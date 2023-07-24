# How to use this code

### Environment dependencies

The following code environments are required: <br>
- Python = 3.9 <br>
- Java \>= SE 6 <br>
- MATLAB \>= r2018a <br>

The following environment variables need to be set: <br>
<br>
`SPIKECOUNTER_PATH` - the base path of the code repository.<br>
`DATA_ROOTDIR` - the base path of the raw data.<br>
`ANALYSIS_OUTPUT_ROOTDIR` - where you want analysis outputs and plots to go.<br>
`DATA_REMOTE_ROOTDIR` - \(Optional\) This was used to access specific datasets from file servers. Replace with `DATA_ROOTDIR` in code.<br>
`CONDA_ENV` - \(Optional\) The name of your conda environment. Only necessary if you intend to use the shell scripts.<br>

### Python package dependencies

All package dependencies are contained in `config/environment.yml`. An environment can be built using [Anaconda](https://docs.conda.io/en/latest/miniconda.html) as follows:<br>

```
conda env create -f config/environment.yml
```

### Getting started

Analysis operations which were iterated over datasets are found in `spikecounter`, and `simulations` for QIF and Morris-Lecar simulations. These were run in parallel on a SLURM cluster, using batch scripting contained in `cluster` subfolders. Further analysis and plotting were performed in Jupyter notebooks, in the `notebooks` subfolder.