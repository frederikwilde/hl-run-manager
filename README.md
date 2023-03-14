# Hamiltonian/Lindbladian learning numerics manager
Contains scripts and modules used for planning, executing, and monitoring numerical experiments.

## Environment
Some variables are necessary. These can either be loaded into the environment, or set via a `config.json` file.
### RESULT_DIR
Results, in particular binary files such as databases and HDF5 files are stored in a separate directory.
The location of this directory must be specified by the `RESULT_DIR` variable.

### DATASET_DIR
The directory which the dataset for learning is found.

### PYTHON_MODULE
Optional. Only applicable for creating Slurm scripts where the specific Python version needs to be loaded as a module.

### VENV_PATH
Optional. For generating Slurm scripts, in which the virtual environment needs to be activated.

## Serialization
Numerical experiments are divided into series which have corresponding folders in the result directory.
Each series contains a collection of runs (corresponding to `Run` objects), each of which corresponds
to one execution of the `execute()` method, i.e. one optimization procedure. Each series corresponds to a particular version of this
package, indicated by its git commit hash in the title.

Each series has a folder in the result directory named as `<number>_<name>_<git-hash>`, where `number`
enumerates all series and `name` is some human-readable information about the particular series.
A series folder is structured as follows:

    001_my-cool-series_45h0f3
    ├── output
    │   ├── <run-id>.log
    │   ├── <run-id>.hdf5
    │   ├── <job-id>.out    (for SLURM jobs)
    │   ...
    ├── scripts
    │   ├── <name>-<run-id>.sh
    │   ├── <name>-<run-id,...,run-id>.sh   (for job arrays on SLURM)
    │   ...
    ├── readme.txt
    └── results.db

## Debugging
In order to run the manager with a dirty git repository set `DEBUG=1` in the environment.