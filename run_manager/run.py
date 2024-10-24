from typing import Tuple, List
from datetime import datetime, UTC
import warnings
import os
from enum import Enum
import re
import subprocess
from sqlalchemy import select
from sqlalchemy.types import Integer, Float, String, DateTime, Text
from sqlalchemy.schema import Column
from pathlib import Path
import logging
import numpy as np
import jax
import jax.numpy as jnp
from time import time
import h5py

from run_manager.series import Series
from run_manager.generate_data import ini_mps
from run_manager import RESULT_DIR, ORMBase, DATASET_DIR


class Status(Enum):
    NOT_IN_DB = "Run needs to be pushed to the database first. Use run.add_to_db()"
    IN_DB = "Run is in the database, but no log file exists yet."
    JOB_STARTED = "log file exists"
    FAILED = "The out file contains an error message."
    FINISHED = "The log file indicates the run finished successfully."


class Run(ORMBase):
    '''
    Representation of one optimization run to store in the results database.

    Saving a Run instance to a database must be done via the `save_to_db` method.

    Args:
        initial_point_seed (int)
        num_sites (int)
        step_size (float)
        deltat (float)
        time_stamps (str): List of indices of the time points to pick in the
            dataset seperated by commas. I.e. `'0,1,5,7'`.
        chi (int)
        local_dim (int)
        num_samples (int): Number of samples for each time point.
        batch_size (int)
        ini_states (str): The initial states for the time evolution, separated by commas.
            Available initial states are documented in `run_manager.generate_data.ini_mps`.
        mps_perturbation (float)
        opt_method (str): Description of the optimizer.
        max_epochs (int)
        data_sets (str): Names of the datasets, separated by commas.
            All datasets much have matching true_params
        series_name (str): The data series, this run is stored in. Get's filled in automatically.
    '''
    __tablename__ = 'Runs'
    id = Column(Integer, primary_key=True)
    initial_point_seed = Column(Integer)
    num_sites = Column(Integer)
    step_size = Column(Float)
    deltat = Column(Float)
    time_stamps = Column(String(50))
    chi = Column(Integer)
    local_dim = Column(Integer)
    num_samples = Column(Integer)
    batch_size = Column(Integer)
    ini_states = Column(String(200))
    mps_perturbation = Column(Float, default=1e-6)
    opt_method = Column(String(20))
    max_epochs = Column(Integer)
    data_sets = Column(String(500))
    time_created = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    series_name = Column(String(100))
    appendix = Column(Text)

    SUCCESS_MESSAGE = 'RUN FINISHED AND SAVED SUCCESSFULLY'

    def _check_consistency(self):
        '''Check consistency of the parameters with the dataset name(s).'''
        # TODO: Ideally, datasets get their own dataclass, which specifies a schema and then
        # all information that is contained in datasets is not redundently loaded into
        # the Run object.
        if f'n{self.num_sites}' not in self.data_sets:
            raise ValueError(
                'The number of sites in the dataset does not match '
                'the specified number os sites.'
            )

        data_sets = [s.strip() for s in self.data_sets.split(',')]
        ini_states = [s.strip() for s in self.ini_states.split(',')]
        if not len(data_sets) == len(ini_states):
            raise ValueError('The numbers of datasets and ini states must be equal')

        true_params = None

        for d, i in zip(data_sets, ini_states):
            if i not in d:
                raise ValueError(
                    f'ini state {i} not in dataset {d}. '
                    'ini_states and data_sets might be out of order.'
                )

            with h5py.File(Path.joinpath(Path(DATASET_DIR), Path(d)), 'r') as f:
                other_true_params = self.get_true_params(f)

            if true_params is None:
                true_params = other_true_params
            else:
                if not jnp.allclose(true_params, other_true_params):
                    raise ValueError(f'Dataset {d} has incompatible true_params.')

    def add_to_db(self, series: Series):
        self._check_consistency()

        if not self.id:
            self.series_name = f'{series.number:03}_{series.name}_{series.hash}'
            series.session.add(self)
            series.session.commit()
        else:
            warnings.warn('Run already saved.')

    @property
    def status(self):
        if not self.id:
            return Status.NOT_IN_DB

        if self.read_log_file():
            if self.SUCCESS_MESSAGE in self.read_log_file():
                return Status.FINISHED

            return Status.JOB_STARTED

        return Status.IN_DB

    def pre_execute_check(self):
        if (not self.id) or (not self.series_name):
            raise ValueError(
                'Run is not properly stored in database. '
                'Must have a valid id and series_name attribute.'
                'Use `run.add_to_db()`.'
            )

    # Methods for directories and files associated with the run.
    @property
    def output_directory(self):
        path = Path.joinpath(Path(RESULT_DIR), Path(self.series_name))
        path = Path.joinpath(path, Path('output'))
        return path

    @property
    def scripts_directory(self):
        path = Path.joinpath(Path(RESULT_DIR), Path(self.series_name))
        path = Path.joinpath(path, Path('scripts'))
        return path

    @property
    def slurm_job_id(self):
        log = self.read_log_file()

        if log is None:
            return None

        # We need MULTILINE, since log is one long string.
        # See https://sethmlarson.dev/regex-$-matches-end-of-string-or-newline for more info.
        matches = re.findall(r'SLURM_JOB_ID=.*$', log, re.MULTILINE)
        if matches:
            return matches[0].split('=')[-1]

    def slurm_stats(self, format=None):
        job_id = self.slurm_job_id

        if job_id is None:
            return None

        if format is None:
            format = "jobname,ncpus,reqmem,avevmsize,elapsed"

        out = subprocess.run([
            "sacct",
            f"--jobs={job_id}",
            "--units=M",
            f"--format={format}"
        ], capture_output=True)

        return out.stdout.decode('utf-8')

    def read_out_file(self):
        warnings.warn('Not implemented yet.')

    def read_log_file(self):
        path = Path.joinpath(self.output_directory, Path(f'{self.id}.log'))

        try:
            with open(path, 'r') as f:
                out = f.read()
        except FileNotFoundError:
            out = None

        return out

    def result_file_path(self):
        return Path.joinpath(self.output_directory, Path(f'{self.id}.hdf5'))

    @staticmethod
    def get(attribute: str, value, series):
        '''
        Helper function to conveniently get individual runs.

        Args:
            attribute (str)
            value: The value to match the attribute by.
            series (run_manager.series.Series): The series, in which to look for.

        Returns:
            Single run, if only one was found, else a list of runs.
        '''
        selector = getattr(Run, attribute) == value

        runs = series.session.scalars(select(Run).where(selector)).all()

        if len(runs) == 1:
            return runs[0]

        if len(runs) > 1:
            return runs

    def execute(
        self,
        loss: callable,
        initialization: callable,
        optimizer,
        launch_script: str,
        slurm_job_id,
        *,
        print_progress: bool = False
    ):
        '''Execute the optimization process and store the results.'''
        self.pre_execute_check()

        if os.environ.get('DEBUG') == '1':
            warnings.warn(f'Executing run {self.id} in DEBUG mode. Repository might be dirty.')

        logging.basicConfig(
            filename=Path.joinpath(self.output_directory, Path(f'{self.id}.log')),
            filemode='w',
            format='%(asctime)s %(name)s %(levelname)s:%(message)s',
            level=logging.DEBUG
        )
        logger = logging.getLogger(__name__)
        logging.getLogger('jax').setLevel(logging.INFO)

        if slurm_job_id:
            logger.debug(f'SLURM_JOB_ID={slurm_job_id}\n')

        data_list = self.load_data()
        ini_states = [s.strip() for s in self.ini_states.split(',')]

        opt = optimizer(initialization(self), self.step_size)
        loss_history, param_history, grad_history = [], [], []
        data_shuffle_rng_seed = self.id
        rng = np.random.default_rng(data_shuffle_rng_seed)
        filename = Path.joinpath(self.output_directory, Path(f'{self.id}.hdf5'))

        for e in range(self.max_epochs):
            for ini_state, data in zip(ini_states, data_list):
                steps, data_indeces, samples_list, true_params = data
                rng.shuffle(data_indeces)
                shape = (len(data_indeces) // self.batch_size, self.batch_size)

                for batch_indeces in data_indeces.reshape(*shape):
                    t1 = time()
                    v, g = jax.value_and_grad(loss)(
                        opt.parameters,
                        self.ini_mps(ini_state, rng=rng),
                        self.deltat,
                        steps,
                        [s[batch_indeces] for s in samples_list],
                        len(samples_list) * self.batch_size
                    )
                    loss_history.append(v)
                    param_history.append(opt.parameters)
                    grad_history.append(g)
                    opt.step(g, e, v)

                    diffs = opt.parameters - true_params
                    J_error = np.abs(diffs[0])
                    U_error = np.abs(diffs[1])
                    mu_avg_error = np.linalg.norm(diffs[2:]) / self.num_sites

                    message = (
                        f'Time: {time()-t1:.2f}s  '
                        f'Errors J: {J_error:.05f} U: {U_error:.05f} mu: {mu_avg_error:.05f}'
                    )
                    logger.debug(message)

            # Save histories after every epoch
            with h5py.File(filename, 'w') as f:
                f.create_dataset('loss_history', data=loss_history)
                f.create_dataset('param_history', data=param_history)
                f.create_dataset('grad_history', data=grad_history)
                f.attrs['data_shuffle_rng_seed'] = data_shuffle_rng_seed
                f.attrs['steps'] = steps
                f.attrs['true_params'] = true_params
                f.attrs['launch_script'] = launch_script
                if os.environ.get('DEBUG') == '1':
                    f.attrs['DEBUG'] = '1'

            message = f'Epoch {e+1} done\n'
            if print_progress:
                print(message[:-1])
            logger.debug(message)

        logger.debug(self.SUCCESS_MESSAGE)

    # AUXILIARY METHODS
    def __repr__(self):
        if self.id:
            return f'<Run {self.id} {self.time_created:%y-%m-%d %H:%M:%S}>'
        else:
            return '<Run object>'

    def add_attributes_to_hdf5(self, file):
        for k, v in self.__dict__.items():
            try:
                file.attrs[k] = v
            except TypeError:
                file.attrs[k] = str(v)

    def copy(self):
        d = dict(self.__dict__)
        d.pop('_sa_instance_state')
        d.pop('time_created')
        d.pop('series_name')
        d.pop('id')
        try:
            d.pop('DEBUG')
        except KeyError:
            pass
        return self.__class__(**d)

    def ini_mps(self, occupation, rng=None):
        return ini_mps(self.num_sites, self.chi, self.mps_perturbation, self.local_dim, occupation, rng)

    def load_data(self) -> List[Tuple]:
        time_stamp_idx = [int(i) for i in self.time_stamps.split(',')]
        data_indeces = np.arange(self.num_samples)

        data_sets = [s.strip() for s in self.data_sets.split(',')]
        dataset_paths = [Path.joinpath(Path(DATASET_DIR), Path(s)) for s in data_sets]

        output = []
        for path in dataset_paths:
            samples_list, times = [], []
            with h5py.File(path, 'r') as f:
                for i in time_stamp_idx:
                    samples_list.append(f[f'samples/t{i}'][:self.num_samples])
                    times.append(f.attrs['times'][i])
                true_params = self.get_true_params(f)

            steps = []
            prev_t = 0.
            for t in times:
                steps.append(int(np.round((t - prev_t) / self.deltat)))
                prev_t = t

            output.append((steps, data_indeces[:], samples_list, true_params))

        return output

    @staticmethod
    def get_true_params(file: h5py.File) -> jnp.ndarray:
        true_params = jnp.concatenate(
            (jnp.array([file.attrs['J'], file.attrs['U']]), file.attrs['mu'])
        )
        return true_params
