# DM Group Assignment

## Usage 

### Conda

Install miniconda (https://docs.conda.io/en/latest/miniconda.html) or anaconda (https://www.anaconda.com/distribution/)

__Create a new conda environment__

    conda create --prefix .conda_env

__Create env from environment file__

    conda env create --file environment.yml --prefix .conda_env

_for better cross platform compatibility install dependencies from history only_

    conda env create --file environment_history.yml --prefix .conda_env

__Update env from environment file__

    conda env update --prefix .conda_env --file environment_history.yml --prune

__Activate env__

    conda activate .conda_env

__Deactivate env__

    conda deactivate

__Search package__

    conda search {package_name}

__Install package__

    conda install {package_name} --prefix .conda_env

__Remove conda package__

    conda remove {package_name} --prefix .conda_env

__Update env packages__

    conda env update --prefix .conda_env --prune

__Create/Update environment file__

    conda env export --prefix .conda_env > environment.yml

For better cross platform compatibility backup history only:

    conda env export --prefix .conda_env --from-history > environment_history.yml

__Note:__ If inside the environment `--prefix .conda_env` can be omitted.

### Jupyter Notebook

__Start Notebook Server__

	jupyter notebook

__Set virtualenv as ipython kernel__

	conda activate .conda_env
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

## Troubleshooting

_MS Python extension for VSCode/Codium is not able to start the conda environment when executing the notebook._
Instead start notebook server as usual and tell Codium to use this server.

_Conda does not find pytorch package when installing from history_
Add `- pytorch` to channels
