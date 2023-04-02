import logging

from build.lib.sketching import optimizer

from . import settings
from . import optimizer
from .datasets import Dataset
from .experiments import (
    CauchySketchingExperiment,
    ObliviousSketchingExperiment,
    SGDExperiment,
    UniformSamplingExperiment,
)

logger = logging.getLogger(settings.LOGGER_NAME)

# Configure logging to write to file
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler("sketching.log")
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

def run_experiments(dataset: Dataset, min_size, max_size, step_size, num_runs, varreg_lambda, add=False):
    """Runs the sketching experiment and creates .csv output files in the experimental-results folder, used for the Jupyter Notebook plots.

    Parameters
    ----------
    dataset : Dataset
        The dataset to use for the experiment and Sketch
    min_size : int
        lower bound of the grid. First sketch size
    max_size : int
        upper bound of the grid. Not necessarily a valid sketch size used.
        Can possibly exceed max_size a single time, in case min_size + [multiple of step_size] < max_size.
    step_size : int
        interval of the grid
    num_runs : int
        Number of replications. Note that a median is taken of useful quantities, so try uneven numbers.
    varreg_lambda : float
        The hyperparameter used for the variance-regularised logistic regression.
    add : bool
        A flag used if the experimental result should be appended to the .csv (True) otherwise overwrite with new .csv (False).
        Useful if one wants to calculate more replications afterwards for a smoother plot.
    """

    # check if results directory exists
    if not settings.RESULTS_DIR.exists():
        settings.RESULTS_DIR.mkdir()

    logger.info("Starting SGD experiment")
    experiment_sgd = SGDExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sgd_varreg{varreg_lambda}.csv",
        num_runs=num_runs,
        optimizer=optimizer.sgd_optimizer(varreg_lambda=varreg_lambda)
    )
    #experiment_sgd.run(parallel=False)

    logger.info("Starting cauchy sketching experiment")
    experiment_cauchy = CauchySketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_cauchy.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        optimizer=optimizer.L1_optimizer(),
    )
    experiment_cauchy.run(parallel=True, add=add)

    logger.info("Starting uniform sampling experiment")
    experiment_uniform = UniformSamplingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_uniform.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        optimizer=optimizer.base_optimizer()
    )
    #experiment_uniform.run(parallel=True, add=add)



    logger.info("Starting sketching experiment1")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_cosketching1_varreg{varreg_lambda}.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=1,
        kyfan_percent=0.25,
        sketchratio=2/3,
        cohensketch=1,
        optimizer=optimizer.varreg_optimizer(varreg_lambda=varreg_lambda)
    )
    #experiment_sketching.run(parallel=False, add=add)

    logger.info("Starting sketching experiment2")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_cosketching2_varreg{varreg_lambda}.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=1,
        kyfan_percent=1,
        sketchratio= 2/3,
        cohensketch=2,
        optimizer=optimizer.L1_optimizer()
    )
    experiment_sketching.run(parallel=False, add=add)

    logger.info("Starting sketching experiment3")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_cosketching5_varreg{varreg_lambda}.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=1,
        kyfan_percent=1,
        sketchratio= 2/3,
        cohensketch=5,
        optimizer=optimizer.L1_optimizer()
    )
    experiment_sketching.run(parallel=False, add=add)

    logger.info("Starting sketching experiment4")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_cosketching10_varreg{varreg_lambda}.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=1,
        kyfan_percent=1,
        sketchratio= 2/3,
        cohensketch=10,
        optimizer=optimizer.L1_optimizer()
    )
    experiment_sketching.run(parallel=False, add=add)

    #logger.info("Starting sketching experiment5")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_cosketching20_varreg{varreg_lambda}.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=1,
        kyfan_percent=1,
        sketchratio=1/3,
        cohensketch=20,
        optimizer=optimizer.L1_optimizer()
    )
    #experiment_sketching.run(parallel=False, add=add)
