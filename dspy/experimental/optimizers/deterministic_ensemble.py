"""Ensemble optimizer and ensembled program."""
import os
import pickle
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

import numpy as np

import dsp
import dspy
from dspy.teleprompt.teleprompt import Teleprompter


class EnsembledProgram(dspy.Module):
    """An ensemble of candidate programs.

    TODO: The EnsembledProgram should imitate the structure of the individual programs.
    (IF they are all compatible).
    This allows compiling with an ensemble program as a (singular) teacher.
    Basically the top majority-compatible trace will end up being used,
    if dspy.majority is the reduce_fn.
    """

    def __init__(self, reduce_fn=None, size=None, deterministic=False, min_acceptable_score: float = -1.0):
        if not deterministic:
            if not reduce_fn:
                raise ValueError("Set deterministic to True or provide a reduce_fn to reduce the outputs.")
            if size is not None and size == 0:
                raise ValueError("If reduce function is specified, ensemble size must be greater than 0.")

        super().__init__()
        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic
        self.min_acceptable_score = min_acceptable_score
        self.programs = None
        # For deterministic ensemble
        self.exampleset_vectors: Any = None
        self.best_pgm_indices: Any = None
        self.best_pgm_scores: Any = None
        # For SentenceTransformersVectorizer
        self.vectorizer = dsp.SentenceTransformersVectorizer()

    def forward(self, **kwargs) -> dspy.Prediction:
        """Runs the ensemble of programs and returns the outputs."""
        outputs = []

        if self.deterministic:
            serialized_input = " | ".join([f"{key}: {value}" for key, value in kwargs.items()])  # pylint: disable=protected-access
            input_example_vector = self.vectorizer([serialized_input]).astype(np.float32)[0]
            scores = np.dot(self.exampleset_vectors, input_example_vector.T).squeeze()

            k = 1  # get just the top matching example
            nearest_example_idx = (scores.argsort()[-k:][::-1])[0]

            best_program_score = self.best_pgm_scores[nearest_example_idx]
            if self.min_acceptable_score > 0.0 and best_program_score < self.min_acceptable_score:
                best_matching_program_index = -1
            else:
                # get the best matching program index for the nearest example
                best_matching_program_index = self.best_pgm_indices[nearest_example_idx]

            if best_matching_program_index != -1:
                outputs = [self.programs[best_matching_program_index](**kwargs)]

        if not outputs:
            size = self.size if self.size and self.size < len(self.programs) else None
            programs = self.programs[:size] if size else self.programs
            outputs = [prog(**kwargs) for prog in programs]

        if len(outputs) == 1:
            return outputs[0]

        if self.reduce_fn:
            return self.reduce_fn(outputs)

        raise ValueError("Set deterministic to True or provide a reduce_fn to reduce the outputs.")

    def save_folder(self, folder_path: str) -> None:
        """Saves the ensembled program to the folder_path."""
        # if the file or folder with the same path exists delete the existing file/folder
        with suppress(FileNotFoundError):
            shutil.rmtree(folder_path)

        # create the folder using path.mkdir(parents=True)
        Path(folder_path).mkdir(parents=True)

        # create a file called deterministic.pkl and serialize self.best_pgm_indices, self.best_pgm_scores,
        # self.exampleset_vectors
        if self.deterministic:
            deterministic_dict = {
                "exampleset_vectors": self.exampleset_vectors,
                "best_pgm_indices": self.best_pgm_indices,
                "best_pgm_scores": self.best_pgm_scores,
            }
            # serialize the deterministic_dict to a file called deterministic.pkl in the folder_path
            # using python pickle format
            with Path.open(Path(folder_path) / "deterministic.pkl", "wb", encoding="UTF-8") as file:
                pickle.dump(deterministic_dict, file)

        # now loop over all the programs in the compiled model and
        # save them to a file called 1.json, 2.json, 3.json, etc.
        for i, program in enumerate(self.programs):
            program.save(Path(folder_path) / f"{i}.json")

    def load_folder(self, folder_path: str, program_class: type, activate_assertions: bool, *args: Any) -> None:
        """Loads the ensembled program from the folder_path."""
        # get the number of .json files in the folder_path
        num_of_programs = len([f for f in os.listdir(folder_path) if f.endswith(".json")])

        self.programs = []
        for i in range(num_of_programs):
            # instantiate the program class
            program = program_class(*args).activate_assertions() if activate_assertions else program_class(*args)
            program.load(Path(folder_path) / f"{i}.json")
            self.programs.append(program)

        # load the deterministic.pkl file if it exists
        if self.deterministic:
            try:
                with Path.open(Path(folder_path) / "deterministic.pkl", "rb", encoding="UTF-8") as file:
                    deterministic_dict = pickle.load(file)
                    self.exampleset_vectors = deterministic_dict["exampleset_vectors"]
                    self.best_pgm_indices = deterministic_dict["best_pgm_indices"]
                    self.best_pgm_scores = deterministic_dict["best_pgm_scores"]
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"deterministic.pkl file not found in the folder path {folder_path}.") from exc


def default_example_serialization_func(example: Any) -> str:
    """Default serialization function for examples."""
    return " | ".join([f"{key}: {value}" for key, value in example.items() if key in example._input_keys])  # pylint: disable=protected-access


class Ensemble(Teleprompter):
    """An ensemble optimizer that optimizes candidate programs."""

    def __init__(
        self,
        *,
        reduce_fn=None,
        size=None,
        deterministic=False,
        min_acceptable_score: float = -1.0,
        metric_func=None,
        example_serialization_func=default_example_serialization_func,
    ):
        """A common reduce_fn is dspy.majority."""
        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic
        self.min_acceptable_score = min_acceptable_score
        self.metric_func = metric_func
        self.example_serialization_func = example_serialization_func.__func__

        if self.deterministic:
            if not self.metric_func:
                raise ValueError("metric_func must be provided for deterministic ensemble.")
            if not self.example_serialization_func:
                raise ValueError("example_serialization_func must be provided for deterministic ensemble.")

    def process_score_matrix(self, program_subscores_lists: list[list[float]]) -> tuple[list[int], list[float]]:
        """Process the score matrix to find the row indices with the maximum value for each column."""
        # Step 1: Put all values in a matrix
        matrix = np.array(program_subscores_lists)

        # Step 2: calculate the max_scores
        max_scores = np.max(matrix, axis=0)
        # print(max_scores)

        # Step 3: For each column, get a row index representing a row with the maximum score for that column
        best_pgm_indices = [np.where(matrix[:, col] >= max_scores[col])[0][0] for col in range(matrix.shape[1])]

        # For each column, create a list of scores corresponding to the pgm_indices for that column
        best_pgm_scores = [matrix[row, col] for col, row in enumerate(best_pgm_indices)]

        # print(f"Best program indices: {best_pgm_indices}")
        # print(f"Best program scores: {best_pgm_scores}")
        return best_pgm_indices, best_pgm_scores

    def compile(
        self,
        candidate_programs: list[Any],
        exampleset: Optional[Any],
    ) -> EnsembledProgram:
        """Compiles the ensemble of programs."""
        ensembled_program = EnsembledProgram(self.reduce_fn, self.size, self.deterministic, self.min_acceptable_score)
        ensembled_program.programs = [x[-1] for x in candidate_programs]

        if self.deterministic:
            if not exampleset:
                raise ValueError("exampleset must be provided for compiling deterministic ensemble.")

            program_subscores_lists = [pgm[1] for pgm in candidate_programs]
            ensembled_program.best_pgm_indices, ensembled_program.best_pgm_scores = self.process_score_matrix(
                program_subscores_lists,
            )
            if len(ensembled_program.best_pgm_indices) != len(exampleset):
                raise ValueError(
                    "The number of examples in exampleset must match the number of examples in candidate programs."
                    "Did you specify a valset in BootstrapRandomFewShot?",
                    "If so, provide that valset to this function as exampleset.",
                    "Otherwise provide the trainset as exampleset.",
                )

            vectorizer = dsp.SentenceTransformersVectorizer()
            serialized_exampleset_inputs = [self.example_serialization_func(example) for example in exampleset]  # pylint: disable=protected-access
            ensembled_program.exampleset_vectors = vectorizer(serialized_exampleset_inputs).astype(np.float32)

            example_scores = []
            for example in exampleset:
                input_example_vector = vectorizer(self.example_serialization_func(example)).astype(np.float32)
                scores = np.dot(ensembled_program.exampleset_vectors, input_example_vector.T).squeeze()

                k = 1  # get just the top matching example
                nearest_example_idx = (scores.argsort()[-k:][::-1])[0]

                # get the best matching programs indices for the nearest example
                best_matching_program_index = ensembled_program.best_pgm_indices[nearest_example_idx]
                pgm = ensembled_program.programs[best_matching_program_index]
                example_input = {key: value for key, value in example.items() if key in example._input_keys}  # pylint: disable=protected-access
                prediction = pgm(**example_input)
                example_scores.append(self.metric_func(example, prediction, trace=None))

            dspy.logger.info(f"Mean score for each example: {example_scores}")
            dspy.logger.info(f"Overall score: {np.mean(example_scores)}")

        return ensembled_program
