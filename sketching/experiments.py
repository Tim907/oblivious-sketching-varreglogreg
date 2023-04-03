import abc
import logging
from time import perf_counter

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import optimizer, settings
from .datasets import Dataset
from .l2s_sampling import l2s_sampling
from .sketch import Sketch
from .cosketch import Cosketch
from .cauchysketch import Cauchysketch

logger = logging.getLogger(settings.LOGGER_NAME)

_rng = np.random.default_rng()


class BaseExperiment(abc.ABC):
    def __init__(
        self,
        num_runs,
        min_size,
        max_size,
        step_size,
        dataset: Dataset,
        results_filename,
        optimizer: optimizer.base_optimizer,
    ):
        self.num_runs = num_runs
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.dataset = dataset
        self.results_filename = results_filename
        self.optimizer = optimizer
        self.varreg_lambda = optimizer.varreg_lambda
        self.optimizer.setDataset(dataset.get_X(), dataset.get_y(), dataset.get_Z())

    @abc.abstractmethod
    def get_reduced_matrix_and_weights(self, config):
        pass

    def get_config_grid(self):
        """
        Returns a list of configurations that are used to run the experiments.
        """
        grid = []
        for size in np.arange(
            start=self.min_size,
            stop=self.max_size + self.step_size,
            step=self.step_size,
        ):
            for run in range(1, self.num_runs + 1):
                grid.append({"run": run, "size": size})

        return grid

    def optimize(self, reduced_matrix, weights, varreg_lambda):
        return optimizer.optimize(Z=reduced_matrix, w=weights, varreg_lambda=varreg_lambda).x

    def run(self, parallel=False, n_jobs=-2, add=False):
        """Runs the experiment with given settings. Can take a few minutes.

        Parameters
        ----------
        parallel : bool
            A flag used if multiple CPU Cores should be used to run different sketch sizes of the grid in parallel.
        n_jobs : int
            The number of CPU cores used. If -1 all are used. For n_jobs = -2, all but one are used. For n_jobs = -3, all but two etc.
        add : bool, optional
            A flag used if the experimental result should be appended to the .csv (True) otherwise overwrite with new .csv (False).
            Useful if one wants to calculate more replications afterwards for a smoother plot.
        """

        Z = self.dataset.get_Z()
        logger.info(f"Variance-regularization parameter: {self.varreg_lambda}")

        beta_opt = self.dataset.get_beta_opt(self.optimizer)
        objective_function = self.optimizer.get_objective_function()
        f_opt = objective_function(beta_opt)

        logger.info("Running experiments...")

        def job_function(cur_config):
            logger.info(f"Current experimental config: {cur_config}")

            start_time = perf_counter()

            reduced_matrix, weights = self.get_reduced_matrix_and_weights(cur_config)
            sampling_time = perf_counter() - start_time

            cur_beta_opt = self.optimizer.optimize(reduced_matrix, weights)
            total_time = perf_counter() - start_time

            cur_ratio = objective_function(cur_beta_opt) / f_opt
            return {
                **cur_config,
                "ratio": cur_ratio,
                "sampling_time_s": sampling_time,
                "total_time_s": total_time,
            }

        if parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(job_function)(cur_config)
                for cur_config in self.get_config_grid()
            )
        else:
            results = [
                job_function(cur_config) for cur_config in self.get_config_grid()
            ]

        logger.info(f"Writing results to {self.results_filename}")

        df = pd.DataFrame(results)
        if not os.path.isfile(self.results_filename) or add is False:
            df.to_csv(self.results_filename, index=False)
        else:
            df.to_csv(self.results_filename, mode="a", header=False, index=False)

        logger.info("Done.")


class UniformSamplingExperiment(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
        optimizer: optimizer.base_optimizer,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer=optimizer,
        )

    def get_reduced_matrix_and_weights(self, config):
        Z = self.dataset.get_Z()
        n = self.dataset.get_n()
        size = config["size"]

        row_indices = _rng.choice(n, size=size, replace=False)
        reduced_matrix = Z[row_indices]
        weights = np.ones(size)

        return reduced_matrix, weights


class CauchySketchingExperiment(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
        optimizer: optimizer.base_optimizer,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer=optimizer
        )

    def get_reduced_matrix_and_weights(self, config):

        Z = self.optimizer.get_Z()
        n = self.dataset.get_n()
        size = config["size"]
        d = Z.shape[1]

        sketch = Cauchysketch(size, n, d)
        for j in range(0, n):
            sketch.insert(Z[j])

        reduced_matrix = sketch.get_reduced_matrix()
        weights_sketch = sketch.get_weights()

        return reduced_matrix, weights_sketch


class ObliviousSketchingExperiment(BaseExperiment):
    """
    WARNING: This implementation is not thread safe!!!
    """

    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
        h_max,
        kyfan_percent,
        sketchratio,
        cohensketch,
        optimizer: optimizer.base_optimizer,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer = optimizer,
        )
        self.h_max = h_max
        self.kyfan_percent = kyfan_percent
        self.sketchratio = sketchratio
        self.cohensketch = cohensketch

    def get_reduced_matrix_and_weights(self, config):
        Z = self.optimizer.get_Z()
        n = self.dataset.get_n()
        d = Z.shape[1]
        size = config["size"]

        # divide by (h_max + 1) + to get one more block for unif sampling
        if self.cohensketch > 1 :
            N2 = max(int(size * self.sketchratio / (self.h_max * self.cohensketch)), 1)
            N = N2 * self.cohensketch
            b = (n / N) ** (1.0 / self.h_max)
            actual_sketch_size = N * self.h_max
            
            unif_block_size = max(size - actual_sketch_size, 1)
            
            sketch = Cosketch(self.h_max, b, N, n, d, self.cohensketch)
            for j in range(0, n):
                sketch.coinsert(Z[j])
            reduced_matrix = sketch.get_reduced_matrix()
            weights_sketch = sketch.get_weights()
        else:
            N = max(int(size * self.sketchratio / self.h_max), 1)
            b = (n / N) ** (1.0 / self.h_max)
            actual_sketch_size = N * self.h_max
            
            unif_block_size = max(size - actual_sketch_size, 1)
            
            sketch = Sketch(self.h_max, b, N, n, d)
            for j in range(0, n):
                sketch.insert(Z[j])
            reduced_matrix = sketch.get_reduced_matrix()
            weights_sketch = sketch.get_weights()

        # do the unif sampling
        rows = _rng.choice(n, unif_block_size, replace=False)
        unif_sample = Z[rows]

        # concat the sketch and the uniform sample
        reduced_matrix = np.vstack([reduced_matrix, unif_sample])

        weights_unif = np.ones(unif_block_size) * n / unif_block_size

        weights = np.concatenate([weights_sketch, weights_unif])
        weights = weights / np.sum(weights)

        self.cur_kyfan_k = int(N * self.kyfan_percent)
        self.cur_kyfan_max_len = actual_sketch_size
        self.cur_kyfan_block_size = N

        return reduced_matrix, weights

    def optimize(self, reduced_matrix, weights, varreg_lambda):
        return optimizer.optimize(
            reduced_matrix,
            weights,
            block_size=self.cur_kyfan_block_size,
            k=self.cur_kyfan_k,
            max_len=self.cur_kyfan_max_len,
            varreg_lambda=varreg_lambda,
        ).x


class L2SExperiment(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
        optimizer: optimizer.base_optimizer,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer=optimizer,
        )

    def get_reduced_matrix_and_weights(self, config):
        Z = self.dataset.get_Z()
        size = config["size"]

        reduced_matrix, weights = l2s_sampling(Z, size=size)

        return reduced_matrix, weights


class SGDExperiment(BaseExperiment):
    def __init__(
        self,
        num_runs,
        dataset: Dataset,
        results_filename,
        optimizer: optimizer.base_optimizer,
    ):
        n = dataset.get_n()
        super().__init__(
            num_runs=num_runs,
            min_size=n,
            max_size=n,
            step_size=0,
            dataset=dataset,
            results_filename=results_filename,
            optimizer=optimizer,
        )

    def get_config_grid(self):
        grid = []

        for run in range(1, self.num_runs + 1):
            grid.append({"run": run})

        return grid

    def get_reduced_matrix_and_weights(self, config):
        # For SGD, no reduction is performed
        return None, None