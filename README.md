# DM Group Assignment

## Videos

- Task 1: https://ucloud.univie.ac.at/index.php/s/lPyG1vT42CeVxdQ
- Task 2: https://ucloud.univie.ac.at/index.php/s/k8YWkXocp2aISVo

## Using the Algorithm

Import `predecon.py` and create a PreDeCon object with the desired parameters. Call the `fit()` method with the dataset as parameter. The predicted labels can be accessed through the `labels` attribute of the PreDeCon object.

## Reports

`reports`

You can run the notebooks used for creating the report-files.

- `task1/Algorithm_Testing.ipynb`
- `task1/IMDb_Tests.ipynb`
- `tudataset/tud_benchmark/EDA.ipynb`
- `task2/classification_experiments.ipynb`
- `tudataset/tud_benchmark/graph_property_prediction.ipynb`

## Additional code

`util`

### Conda

Install miniconda (https://docs.conda.io/en/latest/miniconda.html) or anaconda (https://www.anaconda.com/distribution/)

__Create a new conda environment__

    conda create --prefix .conda_env

__Create env from environment file__

    conda env create --file environment.yml --prefix .conda_env

_for better cross platform compatibility install dependencies from history only_

    conda env create --file environment_history.yml --prefix .conda_env

__Activate env__

    conda activate ./.conda_env

### Jupyter Notebook

__Start Notebook Server__

	jupyter notebook

__Set virtualenv as ipython kernel__

	conda activate ./.conda_env
	ipython kernel install --user --name=uni.dm.ga
	jupyter notebook
	
Select the installed kernel from the drop down

### Torch Geometric

Torch geometric is missing if conda env is installed from environment_history.
Add torch-geometric (cpu, torch.version=1.6.0) by 

    pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
    pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
    pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
    pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
    pip install torch-geometric

### Troubleshooting

_MS Python extension for VSCode/Codium is not able to start the conda environment when executing the notebook._
Instead start notebook server as usual and tell Codium to use this server.

_Conda does not find pytorch package when installing from history_
Add `- pytorch` to channels
