"""Testing the minority report ensemble with a deterministic flag set to true."""
import datasets

import dspy
from dsp.utils.metrics import f1_score
from dspy.evaluate import Evaluate
from dspy.experimental.optimizers.deterministic_ensemble import Ensemble
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.bootstrap import BootstrapFewShot

ds = datasets.load_dataset("b-mc2/sql-create-context")

demos = [dspy.Example(**d).with_inputs("context", "question") for d in ds["train"]]
train, test = demos[:100], demos[100:200]

OPENAI_API_KEY = input("Enter your OPENAI_API_KEY: ")
lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=4000)
dspy.settings.configure(lm=lm)

exact_match = lambda ex, pred, _trace=None: ex.answer == pred.sql_query

program = dspy.TypedPredictor("context, question -> sql_query")


evaluator_exactmatch = Evaluate(
    devset=test,
    num_threads=30,
    metric=exact_match,
    display_progress=False,
)

print("BASELINE:")
print(evaluator_exactmatch(program))

compiled_bootstrapfewshot = BootstrapFewShot(
    metric=exact_match,
    max_rounds=1,
).compile(
    program,
    trainset=train,
)

print("BOOTSTRAPFEWSHOT:")
print(evaluator_exactmatch(compiled_bootstrapfewshot))

NUMOF_CANDIDATE_PROGRAMS = 15

compiled_bootstrapfewshotwithrandomsearch = BootstrapFewShotWithRandomSearch(
    metric=exact_match,
    max_bootstrapped_demos=5,
    max_labeled_demos=5,
    num_candidate_programs=NUMOF_CANDIDATE_PROGRAMS,
    num_threads=NUMOF_CANDIDATE_PROGRAMS,
).compile(
    program,
    trainset=train,
)

teacher_lm = dspy.OpenAI(model="gpt-4-0125-preview", api_key=OPENAI_API_KEY, max_tokens=4000)

compiled_bootstrapfewshotwithrandomsearch_withteacher = BootstrapFewShotWithRandomSearch(
    metric=exact_match,
    teacher_settings={"lm": teacher_lm},
    max_bootstrapped_demos=5,
    max_labeled_demos=5,
    num_candidate_programs=NUMOF_CANDIDATE_PROGRAMS,
    num_threads=NUMOF_CANDIDATE_PROGRAMS,
).compile(
    program,
    teacher=compiled_bootstrapfewshotwithrandomsearch,
    trainset=train,
)

print("BOOTSTRAPFEWSHOTWITHRANDOMSEARCH:")
print(evaluator_exactmatch(compiled_bootstrapfewshotwithrandomsearch))

print("BOOTSTRAPFEWSHOTWITHRANDOMSEARCH_WITHTEACHER:")
print(evaluator_exactmatch(compiled_bootstrapfewshotwithrandomsearch_withteacher))

print("*************NOW SWITCH TO A METRIC THAT USES THE TRACE FLAG:*************************")


def metric_usingtraceflag(ex, pred, trace=None) -> bool:
    """Return true if the predicted answer is correct."""
    if trace is None:  # if we're doing evaluation or optimization
        return f1_score(ex.answer, pred.sql_query)

    return ex.answer == pred.sql_query


evaluator_metricusingtraceflag = Evaluate(
    devset=test,
    num_threads=30,
    metric=metric_usingtraceflag,
    display_progress=False,
)

compiled_bootstrapfewshotwithrandomsearch = BootstrapFewShotWithRandomSearch(
    metric=metric_usingtraceflag,
    max_bootstrapped_demos=5,
    max_labeled_demos=5,
    num_candidate_programs=NUMOF_CANDIDATE_PROGRAMS,
    num_threads=NUMOF_CANDIDATE_PROGRAMS,
).compile(
    program,
    trainset=train,
)

compiled_bootstrapfewshotwithrandomsearch_withteacher = BootstrapFewShotWithRandomSearch(
    metric=metric_usingtraceflag,
    teacher_settings={"lm": teacher_lm},
    max_bootstrapped_demos=5,
    max_labeled_demos=5,
    num_candidate_programs=NUMOF_CANDIDATE_PROGRAMS,
    num_threads=NUMOF_CANDIDATE_PROGRAMS,
).compile(
    program,
    teacher=compiled_bootstrapfewshotwithrandomsearch,
    trainset=train,
)

print("BOOTSTRAPFEWSHOTWITHRANDOMSEARCH:")
print(evaluator_metricusingtraceflag(compiled_bootstrapfewshotwithrandomsearch))

print("BOOTSTRAPFEWSHOTWITHRANDOMSEARCH_WITHTEACHER:")
print(evaluator_metricusingtraceflag(compiled_bootstrapfewshotwithrandomsearch_withteacher))

print("*************EVALUATING THE PROGRAMS WITH THE ORIGINAL BINARY METRIC:**************")
print("FINAL_BOOTSTRAPFEWSHOTWITHRANDOMSEARCH_WITHTEACHER_WITH_ORIGINAL_METRIC:")
print(evaluator_exactmatch(compiled_bootstrapfewshotwithrandomsearch_withteacher))

compiled_deterministic_ensemble = Ensemble(
    reduce_fn=dspy.majority,
    deterministic=True,
    metric_func=metric_usingtraceflag,
).compile(
    compiled_bootstrapfewshotwithrandomsearch_withteacher.candidate_programs,
    exampleset=train,
)

print("DETERMINISTIC ENSEMBLE OPTIMIZER:")
print(evaluator_metricusingtraceflag(compiled_deterministic_ensemble))

print("DETERMINISTIC ENSEMBLE OPTIMIZER WITH SIZE AND MIN ACCEPTABLE SCORE:")
compiled_deterministic_ensemble.min_acceptable_score = 0.7
compiled_deterministic_ensemble.size = NUMOF_CANDIDATE_PROGRAMS
print(evaluator_metricusingtraceflag(compiled_deterministic_ensemble))
