# Lindbladian learning numerics package
Contains scripts and modules used for planning, executing, and administrating numerical experiments.

## Results directory
Results, in particular binary files such as databases and HDF5 files are stored in a separate directory.
The location of this directory must be specified by the `RESULT_DIR` variable.

## Serialization
Numerical experiments are divided into series which have corresponding folders in the result directory.
Each series contains a collection of runs (corresponding to `Run` objects), each of which corresponds
to one execution of the `main.py` script. Each series corresponds to one particular version of this
package, indicated by its git commit hash in the title.

Each series has a folder in the result directory named as `<number>_<name>_<git-hash>`, where `number`
enumerates all series and `name` is some human-readable information about the particular series.
A series folder is structured as follows:

    001_some-tests_45h0f3
    ├── output
    │   ├── <run-id>.log
    │   ├── <run-id>.hdf5
    │   ├── <job-id>.out    (for SLURM jobs)
    │   ...
    ├── scripts
    │   ├── <run-id>.sh
    │   ├── <array-id>.sh   (for job arrays on SLURM)
    │   ...
    ├── readme.txt
    └── results.db
