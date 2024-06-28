"""Testing the minority report ensemble with a deterministic flag set to true."""
import os

import datasets

import dspy
from dsp.utils.metrics import f1_score
from dspy.evaluate import Evaluate
from dspy.experimental.optimizers.mixture_of_experts import MixtureOfExperts

os.environ["TOKENIZERS_PARALLELISM"] = "true"

ds = datasets.load_dataset("b-mc2/sql-create-context")

demos = [dspy.Example(**d).with_inputs("context", "question") for d in ds["train"]]
train, test = demos[:100], demos[100:200]

OPENAI_API_KEY = input("Enter your OPENAI_API_KEY: ")
lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=4000)
dspy.settings.configure(lm=lm)

exact_match = lambda ex, pred, _trace=None: ex.answer == pred.sql_query

program = dspy.TypedPredictor("context, question -> sql_query")

teacher_lm = dspy.OpenAI(model="gpt-4-0125-preview", api_key=OPENAI_API_KEY, max_tokens=4000)


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

mixture_of_experts = MixtureOfExperts(
    number_of_experts=5,
    metric_func=metric_usingtraceflag,
    teacher_settings={"lm": teacher_lm},
).compile(
    program,
    trainset=train,
)

print("MIXTURE OF EXPERTS WITH DEFAULT CLUSTER FUNCTION:")
print(evaluator_metricusingtraceflag(mixture_of_experts))
