from sketching.datasets import Covertype_Sklearn, NoisyDataset
from sketching.utils import run_experiments

MIN_SIZE = 1000
MAX_SIZE = 20000
STEP_SIZE = 1000
NUM_RUNS = 21

dataset_noisy = NoisyDataset(dataset=Covertype_Sklearn(), percentage=0.01, std=10)

run_experiments(
    dataset=dataset_noisy,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    varreg_lambda=0
)
