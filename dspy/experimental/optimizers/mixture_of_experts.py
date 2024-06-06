"""Mixture of experts."""
import os
import pickle
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import dsp
import dspy
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.teleprompt import Teleprompter


class MixtureOfExpertsProgram(dspy.Module):
    """A mixture of expert candidate programs."""

    def __init__(self):
        super().__init__()
        self.programs = None
        self.classifier = None
        self.classifier_input_func = None
        self.label_encoder = None
        # For SentenceTransformersVectorizer
        self.vectorizer = dsp.SentenceTransformersVectorizer()

    def forward(self, **kwargs) -> dspy.Prediction:
        """Runs the query through the mixture of experts and returns the output."""
        example = dspy.Example(kwargs)
        example_embedding = self.vectorizer([self.classifier_input_func(example)]).astype(np.float32)
        predicted_cluster_index = self.classifier.predict(example_embedding)
        predicted_program = self.programs[self.label_encoder.inverse_transform(predicted_cluster_index)[0]]
        return predicted_program(**kwargs)

    def save_folder(self, folder_path: str) -> None:
        """Saves the ensembled program to the folder_path."""
        # if the file or folder with the same path exists delete the existing file/folder
        with suppress(FileNotFoundError):
            shutil.rmtree(folder_path)

        # create the folder using path.mkdir(parents=True)
        Path(folder_path).mkdir(parents=True)

        # create a file called mixture_of_experts.pkl and serialize self.classifier,
        # self.classifier_input_func, self.label_encoder
        moe_dict = {
            "classifier": self.classifier,
            "classifier_input_func": self.classifier_input_func,
            "label_encoder": self.label_encoder,
        }
        # serialize the moe_dict to a file called mixture_of_experts.pkl in the folder_path
        # using python pickle format
        with Path.open(Path(folder_path) / "mixture_of_experts.pkl", "wb", encoding="UTF-8") as file:
            pickle.dump(moe_dict, file)

        # now loop over all the programs in the compiled model and
        # save them to a file called 1.json, 2.json, 3.json, etc.
        for i, program in enumerate(self.programs):
            program.save(Path(folder_path) / f"{i}.json")

    def load_folder(self, folder_path: str, program_class: type, activate_assertions: bool, *args: Any) -> None:
        """Loads the mixture of experts program from the folder_path."""
        # get the number of .json files in the folder_path
        num_of_programs = len([f for f in os.listdir(folder_path) if f.endswith(".json")])

        self.programs = []
        for i in range(num_of_programs):
            # instantiate the program class
            program = program_class(*args).activate_assertions() if activate_assertions else program_class(*args)
            program.load(Path(folder_path) / f"{i}.json")
            self.programs.append(program)

        # load the mixture_of_experts.pkl file if it exists
        try:
            with Path.open(Path(folder_path) / "mixture_of_experts.pkl", "rb", encoding="UTF-8") as file:
                moe_dict = pickle.load(file)
                self.classifier = moe_dict["classifier"]
                self.classifier_input_func = moe_dict["classifier_input_func"]
                self.label_encoder = moe_dict["label_encoder"]
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"mixture_of_experts.pkl file not found in the folder path {folder_path}.") from exc


def default_cluster_func(examples: list[dspy.Example], num_of_clusters) -> list[list[dspy.Example]]:
    """Default cluster function clusters on output keys."""
    serialized_outputs = [
        " | ".join([f"{key}: {value}" for key, value in example.items() if key not in example._input_keys])
        for example in examples
    ]

    vectorizer = dsp.SentenceTransformersVectorizer()
    trainset_vectors = vectorizer(serialized_outputs).astype(np.float32)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=42)
    cluster_indices = kmeans.fit_predict(trainset_vectors)

    example_clusters = []
    for cluster in range(num_of_clusters):
        example_clusters.append([examples[i] for i in range(len(examples)) if cluster_indices[i] == cluster])

    return example_clusters


def default_classifier_input_func(example: dspy.Example) -> str:
    """Default classifier input function just serializes the input keys."""
    return " | ".join([f"{key}: {value}" for key, value in example.items() if key in example._input_keys])  # pylint: disable=protected-access


class MixtureOfExperts(Teleprompter):
    """An ensemble optimizer that optimizes candidate programs."""

    def __init__(
        self,
        *,
        number_of_experts: int,
        metric_func: Any,
        cluster_func=default_cluster_func,
        classifier_input_func=default_classifier_input_func,
        bootstrapfewshotwithrandomsearch_teacher_settings=None,
        bootstrapfewshotwithrandomsearch_max_bootstrapped_demos=5,
        bootstrapfewshotwithrandomsearch_max_labeled_demos=5,
        bootstrapfewshotwithrandomsearch_num_candidate_programs=3,
        bootstrapfewshotwithrandomsearch_num_threads=3,
    ):
        """A common reduce_fn is dspy.majority."""
        self.number_of_experts = number_of_experts
        self.metric_func = metric_func
        self.cluster_func = cluster_func.__func__
        self.classifier_input_func = classifier_input_func.__func__

        self.bootstrapfewshotwithrandomsearch_teacher_settings = (bootstrapfewshotwithrandomsearch_teacher_settings,)
        self.bootstrapfewshotwithrandomsearch_max_bootstrapped_demos = (
            bootstrapfewshotwithrandomsearch_max_bootstrapped_demos,
        )
        self.bootstrapfewshotwithrandomsearch_max_labeled_demos = (bootstrapfewshotwithrandomsearch_max_labeled_demos,)
        self.bootstrapfewshotwithrandomsearch_num_candidate_programs = (
            bootstrapfewshotwithrandomsearch_num_candidate_programs,
        )
        self.bootstrapfewshotwithrandomsearch_num_threads = (bootstrapfewshotwithrandomsearch_num_threads,)

    def compile(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        valset: Optional[list[dspy.Example]],
    ) -> MixtureOfExpertsProgram:
        """Compiles the mixture of experts."""
        if not trainset:
            raise ValueError("trainset must be provided for compiling the mixture of experts.")

        moe_program = MixtureOfExpertsProgram()

        # call cluster_func with the trainset to get back trainset clusters
        clusters_of_examples = self.cluster_func(trainset, self.number_of_experts)

        # optimize each trainset cluster with bootstraprandomfewshow and
        # build a collection of best candidate programs for each cluster
        candidate_programs = []
        for example_cluster in clusters_of_examples:
            compiled_bootstrapfewshotwithrandomsearch = BootstrapFewShotWithRandomSearch(
                metric=self.metric_func,
                max_bootstrapped_demos=self.bootstrapfewshotwithrandomsearch_max_bootstrapped_demos,
                max_labeled_demos=self.bootstrapfewshotwithrandomsearch_max_labeled_demos,
                num_candidate_programs=self.bootstrapfewshotwithrandomsearch_num_candidate_programs,
                num_threads=self.bootstrapfewshotwithrandomsearch_num_threads,
            ).compile(
                program,
                trainset=example_cluster,
            )

            compiled_bootstrapfewshotwithrandomsearch_withteacher = BootstrapFewShotWithRandomSearch(
                metric=self.metric_func,
                teacher_settings=(
                    self.bootstrapfewshotwithrandomsearch_teacher_settings
                    if self.bootstrapfewshotwithrandomsearch_teacher_settings
                    else None
                ),
                max_bootstrapped_demos=self.bootstrapfewshotwithrandomsearch_max_bootstrapped_demos,
                max_labeled_demos=self.bootstrapfewshotwithrandomsearch_max_labeled_demos,
                num_candidate_programs=self.bootstrapfewshotwithrandomsearch_num_candidate_programs,
                num_threads=self.bootstrapfewshotwithrandomsearch_num_threads,
            ).compile(
                program,
                teacher=compiled_bootstrapfewshotwithrandomsearch,
                trainset=example_cluster,
            )

            best_candidate_program_info = compiled_bootstrapfewshotwithrandomsearch_withteacher.candidate_programs[0]
            candidate_programs.append(best_candidate_program_info[-1])  # this is the best candidate program

        # train a LinearSVM classifier to predict the best candidate program
        # x is trainset examples from a cluster passed through the classifier_input_func and then tokenized
        # y is the cluster index (cluster index also points to the best candidate program in candidate_programs list)
        x_values = []
        y_values = []
        for cluster_index, example_cluster in enumerate(clusters_of_examples):
            for example in example_cluster:
                serialized_input = self.classifier_input_func(example)
                x_values.append(serialized_input)
                y_values.append(cluster_index)

        # Convert x_values to embeddings
        vectorizer = dsp.SentenceTransformersVectorizer()
        x_embeddings = vectorizer(x_values).astype(np.float32)

        # Encode the labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_values)

        # Create and train the LinearSVM classifier
        classifier = svm.SVC()
        classifier.fit(x_embeddings, y_encoded)

        # initialize exampleset as valset if its provided otherwise trainset
        exampleset = valset if valset else trainset

        # test the mixture of experts and log/report the results
        example_scores = []
        for example in exampleset:
            example_embedding = vectorizer([self.classifier_input_func(example)]).astype(np.float32)
            predicted_cluster_index = classifier.predict(example_embedding)
            predicted_program = candidate_programs[label_encoder.inverse_transform(predicted_cluster_index)[0]]
            example_input_dict = {key: value for key, value in example.items() if key in example._input_keys}  # pylint: disable=protected-access
            prediction = predicted_program(**example_input_dict)
            example_scores.append(self.metric_func(example, prediction, trace=None))

        dspy.logger.info(f"Mean score for each example: {example_scores}")
        dspy.logger.info(f"Overall score: {np.mean(example_scores)}")

        moe_program.programs = candidate_programs
        moe_program.classifier = classifier
        moe_program.classifier_input_func = self.classifier_input_func
        moe_program.label_encoder = label_encoder

        return moe_program
