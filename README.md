# Almost Linear Constant-Factor Sketching for 𝓁₁ and Logistic Regression 

[![python-version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)

This is the accompanying code repository for the ICLR 2023 publication "Almost Linear Constant-Factor Sketching for 𝓁₁ and Logistic Regression" by Alexander Munteanu, Simon Omlor and David P. Woodruff. 

## How to install

1. Clone the repository and navigate into the new directory

   ```bash
   git clone https://github.com/Tim907/oblivious_sketching_varreglogreg
   cd oblivious_sketching_varreglogreg
   ```

2. Create and activate a new virtual environment
   
   on Unix:
   ```bash
   python -m venv venv
   . ./venv/bin/activate
   ```
   on Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate.bat
   ```

3. Install the package locally

   ```bash
   pip install .
   ```

4. To confirm that everything worked, install `pytest` and run the tests
   ```bash
   pip install pytest
   python -m pytest
   ```

## How to run the experiments

The `scripts` directory contains multiple python scripts that can be
used to run the experiments.
Just make sure, that everything is installed properly.

For example, to run the covertype experiments you can use the following command:

```bash
python scripts/run_experiments_covertype.py
```

You can try different optimizers for the experiments, by changing the experiments in `utils.run_experiments` to the classes defined in `optimizer.py`.
There are optimizers for logistic likelihood, variance-regularized logistic likelihood, L1-optimization and Stochastic gradient descent

## How to recreate the plots

The plots can be recreated using the jupyter notebooks that can be
found in the `notebooks` directory.
Instructions on how to set up a jupyter environment can be found
[here](https://jupyter.org/).
