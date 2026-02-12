from typing import Tuple, List
from datetime import datetime, UTC
import warnings
import os
from enum import Enum
import re
import subprocess
from sqlalchemy import select
from sqlalchemy.types import Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.schema import Column
from pathlib import Path
import logging
import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from time import time, sleep
import h5py
from sqlalchemy.exc import OperationalError

from run_manager.series import Series
from run_manager.generate_data import ini_mps, DataSet
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
        step_size (float)
        deltat (float)
        time_stamps (str): List of indices of the time points to pick in the
            dataset seperated by commas. I.e. `'0,1,5,7'`.
        chi (int)
        local_dim (int)
        num_samples (int): Number of samples for each time point.
        batch_size (int)
        mps_perturbation (float)
        max_epochs (int)
        data_sets (str): Names of the datasets, separated by commas.
            All datasets much have matching true_params
        parity_project (bool): Loads the data and projects it to parity measurements.
            Also instructs the loss function to parity project the mps.
        appendix (str): Additional information.

        # Arguments below get filled automatically by commiting the run to the database via
        # save_to_db()

        time_created (DateTime)
        series_name (str): The data series, this run is stored in. Get's filled in automatically.
        num_sites (int)
        ini_states (str): The initial states for the time evolution, separated by commas.
            Available initial states are documented in `run_manager.generate_data.ini_mps`.
    '''
    __tablename__ = 'Runs'
    id = Column(Integer, primary_key=True)
    initial_point_seed = Column(Integer)
    initial_point_bias = Column(Float)
    initial_point_stddev = Column(Float)
    step_size = Column(Float)
    deltat = Column(Float)
    time_stamps = Column(String(50))
    chi = Column(Integer)
    local_dim = Column(Integer)
    num_samples = Column(Integer)
    batch_size = Column(Integer)
    mps_perturbation = Column(Float, default=1e-6)
    max_epochs = Column(Integer)
    data_sets = Column(String(500))
    parity_project = Column(Boolean, nullable=False)
    bfgs_gtol = Column(Float, default=1e-4)
    bfgs_maxiter = Column(Integer, default=100)
    appendix = Column(Text)

    # Fields that get filled automatically.
    time_created = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    num_sites = Column(Integer)
    series_name = Column(String(100))
    ini_states = Column(String(200))
    true_params = Column(String(500))

    SUCCESS_MESSAGE = 'RUN FINISHED AND SAVED SUCCESSFULLY'

    def _check_consistency_and_set_vars(self):
        '''Check consistency of the parameters with the dataset name(s).'''
        # TODO: Ideally, datasets get their own dataclass, which specifies a schema and then
        # all information that is contained in datasets is not redundently loaded into
        # the Run object.

        # check num_sites and true_params
        num_sites = None
        true_params = None
        ini_states = []
        for path in self.dataset_paths:
            with h5py.File(path, 'r') as f:
                ini_states.append(f.attrs['ini_state'])

                if num_sites is None:
                    num_sites = int(f.attrs['num_sites'])  # np.int64 gets converted to bytes
                    true_params = f.attrs['true_parameters']
                else:
                    if num_sites != f.attrs['num_sites']:
                        raise ValueError(
                            'num_sites must be the same in all datasets'
                        )
                    if not jnp.allclose(true_params, f.attrs['true_parameters']):
                        raise ValueError(
                            'True parameters must be equal in all datasets'
                        )

        # set ini_states, num_sites
        self.num_sites = num_sites
        self.ini_states = ','.join(ini_states)
        self.true_params = ','.join([f'{x:.2f}' for x in true_params])

    def add_to_db(self, series: Series):
        self._check_consistency_and_set_vars()

        if not self.id:
            self.series_name = f'{series.number:03}_{series.name}_{series.hash}'
            series.session.add(self)
            series.session.commit()
        else:
            warnings.warn('Run already saved.')

    def get_true_params_from_dataset(self):
        '''The true_params attribute is a string and only meant for printing.'''
        dataset = self.data_sets.split(',')[0].strip()
        dataset = DataSet.from_hdf5(dataset, num_samples=1)
        return dataset.true_parameters

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
    def dataset_paths(self):
        data_sets = [s.strip() for s in self.data_sets.split(',')]
        return [Path.joinpath(Path(DATASET_DIR), Path(s)) for s in data_sets]

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

    @property
    def debug_mode(self):
        log = self.read_log_file()

        if log is None:
            return None

        matches = re.findall('DEBUG=1', log)
        if matches:
            return True

        return False

    @property
    def total_time_from_logfile(self):
        logfile = self.read_log_file()

        if logfile is None:
            return None

        matches = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', logfile)

        if len(matches) >= 2:
            return datetime.strptime(matches[-1], r'%Y-%m-%d %H:%M:%S') - datetime.strptime(matches[0], r'%Y-%m-%d %H:%M:%S')

        return 'n/a'

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
        logger = self.__setup_logging()

        if os.environ.get('DEBUG') == '1':
            warnings.warn(f'Executing run {self.id} in DEBUG mode. Repository might be dirty.')
            logger.debug('DEBUG=1')

        if slurm_job_id:
            logger.debug(f'SLURM_JOB_ID={slurm_job_id}\n')

        data_list = self.load_data()

        opt = optimizer(initialization(self), self.step_size)
        loss_history, param_history, grad_history = [], [], []
        data_shuffle_rng_seed = self.id
        rng = np.random.default_rng(data_shuffle_rng_seed)
        filename = Path.joinpath(self.output_directory, Path(f'{self.id}.hdf5'))

        data_indeces = np.arange(self.num_samples)

        for e in range(self.max_epochs):
            rng.shuffle(data_indeces)
            shape = (len(data_indeces) // self.batch_size, self.batch_size)

            for batch_indeces in data_indeces.reshape(*shape):
                t1 = time()
                v, g = None, None

                for data in data_list:
                    steps, _, samples_list, true_params, ini_state = data

                    _v, _g = jax.value_and_grad(loss)(
                        opt.parameters,
                        self.ini_mps(ini_state, rng=rng),
                        self.deltat,
                        steps,
                        [s[batch_indeces] for s in samples_list],
                        len(samples_list) * self.batch_size,
                        self.parity_project
                    )
                    v = _v if v is None else v + _v
                    g = _g if g is None else g + _g

                loss_history.append(v)
                param_history.append(opt.parameters)
                grad_history.append(g)
                opt.step(jnp.clip(g, -20, 20), e, v)  # for temporarily fixing parameters: .at[slice].set(0)

                diffs = opt.parameters - true_params
                J1_error = np.abs(diffs[0])
                J2_error = np.abs(diffs[1])
                U_error = np.abs(diffs[2])
                mu_avg_error = np.linalg.norm(diffs[3:]) / self.num_sites

                message = (
                    f'Time: {time()-t1:.2f}s  '
                    f'Errors J1: {J1_error:.05f} J2: {J2_error:.05f} U: {U_error:.05f} mu: {mu_avg_error:.05f}'
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
        logger.debug('Starting second optimizer')

        def value_and_grad_recorded(params):
            '''Helper to record calls from scipy.optimize.minimize into histories.'''
            v, g = jax.value_and_grad(loss)(
                params,
                self.ini_mps(ini_state, rng=rng),
                self.deltat,
                steps,
                samples_list,
                len(samples_list) * self.num_samples,
                self.parity_project
            )
            loss_history.append(v)
            param_history.append(params)
            grad_history.append(g)
            return v, g

        result = minimize(
            value_and_grad_recorded,
            opt.parameters,
            method='BFGS',
            jac=True,
            options={'gtol': self.bfgs_gtol, 'maxiter': self.bfgs_maxiter},
        )
        logger.debug(result.get('message'))
        v, g = value_and_grad_recorded(result['x'])  # record the solution

        with h5py.File(filename, 'w') as f:
            f.create_dataset('loss_history', data=loss_history)
            f.create_dataset('param_history', data=param_history)
            f.create_dataset('grad_history', data=grad_history)
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

        output = []
        for path in self.dataset_paths:
            dataset = DataSet.from_hdf5(
                path,
                time_stamp_selection=time_stamp_idx,
                num_samples=self.num_samples,
                parity_project=self.parity_project
            )

            steps = []
            prev_t = 0.
            for t in dataset.times:
                steps.append(int(np.round((t - prev_t) / self.deltat)))
                prev_t = t

            output.append((
                steps,
                data_indeces.copy(),
                dataset.samples_list,
                dataset.true_parameters,
                dataset.ini_state
            ))

        return output

    def __setup_logging(self):
        logging.basicConfig(
            filename=Path.joinpath(self.output_directory, Path(f'{self.id}.log')),
            filemode='w',
            format='%(asctime)s %(name)s %(levelname)s:%(message)s',
            level=logging.DEBUG
        )
        logging.getLogger('jax').setLevel(logging.INFO)
        return logging.getLogger(__name__)
    

def safe_query(series, run_index: int) -> Run:
    # Obtain the Run object from the database. This might lead to collisions,
    # when other processes are querying as well. I don't think this is the fault
    # of SQLite or SQLAlchemy, but rather the filesystem.
    # Some info on concurrency can be found here: https://sqlite.com/faq.html#q5
    t = time()

    while time() - t <= 60:  # timeout after 60s
        try:
            run = series.session.query(Run).where(Run.id == run_index).first()
            return run

        except OperationalError:
            sleep(np.random.rand())

    raise TimeoutError('Querying database was unsuccessful.')