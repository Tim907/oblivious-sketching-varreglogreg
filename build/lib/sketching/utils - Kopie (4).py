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
        kyfan_percent=0.25,
        sketchratio=2/3,
        cohensketch=7 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment4")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching4.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=0.25,
        sketchratio= 0.5,
        cohensketch=7 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment5")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching5.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=1,
        sketchratio=2/3,
        cohensketch=1 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment6")
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

    logger.info("Starting sketching experiment7")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching7.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=1,
        sketchratio=2/3,
        cohensketch=7 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment8")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching8.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=1,
        sketchratio= 0.5,
        cohensketch=7 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment9")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching9.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=0.25,
        sketchratio=1/3,
        cohensketch=1 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment10")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching10.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=0.6,
        sketchratio= 0.5,
        cohensketch=1 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment11")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching11.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=0.25,
        sketchratio=1/3,
        cohensketch=3 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment12")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching12.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=0.5,
        sketchratio= 0.5,
        cohensketch=3 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment13")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching13.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=1,
        sketchratio=1/3,
        cohensketch=1 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment14")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching14.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=1,
        sketchratio= 0.2,
        cohensketch=1 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment15")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching15.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=1,
        sketchratio=2/3,
        cohensketch=3 
    )
    experiment_sketching.run(parallel=False)

    logger.info("Starting sketching experiment16")
    experiment_sketching = ObliviousSketchingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_sketching16.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        h_max=2,
        kyfan_percent=1,
        sketchratio= 0.5,
        cohensketch=3 
    )
    experiment_sketching.run(parallel=False)
