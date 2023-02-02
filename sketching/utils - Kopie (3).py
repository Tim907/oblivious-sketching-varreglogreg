import logging

from . import settings
from .datasets import Dataset
from .experiments import (
    ObliviousSketchingExperiment,
    UniformSamplingExperiment,
)

logger = logging.getLogger(settings.LOGGER_NAME)


def run_experiments(dataset: Dataset, min_size, max_size, step_size, num_runs):
    # check if results directory exists
    if not settings.RESULTS_DIR.exists():
        settings.RESULTS_DIR.mkdir()




    logger.info("Starting uniform sampling experiment")
    experiment_uniform = UniformSamplingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_uniform.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
    )
    experiment_uniform.run(parallel=True)



    logger.info("Starting sketching experiment1")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching1.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=0.25,
        sketchratio=2/3,
        cohensketch=1 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment2")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching2.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=0.25,
        sketchratio= 0.5,
        cohensketch=1 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment3")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching3.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=0.75,
        sketchratio=0.5,
        cohensketch=1 
    )
    experiment_sketching.run(parallel=False)


    logger.info("Starting sketching experiment4")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching6.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=1,
        sketchratio= 0.5,
        cohensketch=1 
    )
    experiment_sketching.run(parallel=False)
