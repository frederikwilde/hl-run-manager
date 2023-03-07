from datetime import datetime
import warnings
import os
from sqlalchemy.types import Integer, Float, String, DateTime, Text
from sqlalchemy.schema import Column
from pathlib import Path
import logging
import numpy as np
import jax
import jax.numpy as jnp
from time import time
import h5py

from .series import Series
from .generate_data import ini_mps
from . import RESULT_DIR
from . import ORMBase
from . import DATASET_DIR


class Run(ORMBase):
    '''
    Representation of one optimization run to store in the results database.

    Saving a Run instance to a database must be done via the `save_to_db` method.

    The `status` method can return the following values:
        NOT_IN_DB
        IN_DB
        IN_SLURM_SCRIPT
        JOB_STARTED: log file exists
        FAILED: out file contains error message
        FINISHED: log file contains confirmation that the HDF5 file as been created successfully

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
        mps_perturbation (float)
        opt_method (str): Description of the optimizer.
        max_epochs (int)
        data_set (str)
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
    mps_perturbation = Column(Float, default=1e-6)
    opt_method = Column(String(20))
    max_epochs = Column(Integer)
    data_set = Column(String(100))
    time_created = Column(DateTime, nullable=False, default=datetime.utcnow)
    series_name = Column(String(100))
    appendix = Column(Text)

    SUCCESS_MESSAGE = 'RUN FINISHED AND SAVED SUCCESSFULLY'

    def add_to_db(self, series: Series):
        if not self.id:
            self.series_name = f'{series.number:03}_{series.name}_{series.hash}'
            series.session.add(self)
            series.session.commit()
        else:
            warnings.warn('Run already saved.')

    @property
    def status(self):
        warnings.warn('Not entirely implemented yet.')
        if not self.id:
            out = 'NOT_IN_DB'
        else:
            out = 'IN_DB'
            if self.read_log_file():
                if self.SUCCESS_MESSAGE in self.read_log_file():
                    out = 'FINISHED'
                else:
                    out = 'JOB_STARTED'

        return out

    def pre_execute_check(self):
        if (not self.id) or (not self.series_name):
            raise ValueError('Run is not properly stored in database. Must have a valid id and series_name attribute.')

    # Methods for directories and files associated with the run.
    def output_directory(self):
        path = Path.joinpath(Path(RESULT_DIR), Path(self.series_name))
        path = Path.joinpath(path, Path('output'))
        return path
    
    def scripts_directory(self):
        path = Path.joinpath(Path(RESULT_DIR), Path(self.series_name))
        path = Path.joinpath(path, Path('scripts'))
        return path

    def find_job_id(self):
        warnings.warn('Not implemented yet.')
        for f in os.scandir(self.output_directory()):
            if f.name[-4:] == '.out':
                pass

    def read_out_file(self):
        warnings.warn('Not implemented yet.')

    def read_log_file(self):
        path = Path.joinpath(self.output_directory(), Path(f'{self.id}.log'))

        try:
            with open(path, 'r') as f:
                out = f.read()
        except FileNotFoundError:
            out = None
        
        return out

    def result_file_path(self):
        return Path.joinpath(self.output_directory(), Path(f'{self.id}.hdf5'))

    def execute(
        self,
        loss: callable,
        initialization: callable,
        optimizer,
        launch_script: str
    ):
        '''Execute the optimization process and store the results.'''
        self.pre_execute_check()
        if os.environ.get('DEBUG') == '1':
            warnings.warn(f'Executing run {self.id} in DEBUG mode. Repository might be dirty.')

        logging.basicConfig(
            filename=Path.joinpath(self.output_directory(), Path(f'{self.id}.log')),
            filemode='w',
            format='%(asctime)s %(levelname)s:%(message)s',
            level=logging.DEBUG
        )

        steps, data_indeces, samples_list, true_params = self.load_data()

        opt = optimizer(initialization(self), self.step_size)
        loss_history, param_history, grad_history = [], [], []
        data_shuffle_rng_seed = self.id
        rng = np.random.default_rng(data_shuffle_rng_seed)
        filename = Path.joinpath(self.output_directory(), Path(f'{self.id}.hdf5'))

        for e in range(self.max_epochs):
            rng.shuffle(data_indeces)
            for batch_indeces in data_indeces.reshape(len(data_indeces)//self.batch_size, self.batch_size):
                t1 = time()
                v, g = jax.value_and_grad(loss)(
                    opt.parameters,
                    self.ini_mps('neel', rng=rng),
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
                print(message)
                logging.debug(message)

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
            print(message)
            logging.debug(message)

        logging.debug(self.SUCCESS_MESSAGE)

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

    def load_data(self):
        time_stamp_idx = [int(i) for i in self.time_stamps.split(',')]
        dataset_path = Path.joinpath(Path(DATASET_DIR), Path(self.data_set))

        samples_list, times = [], []
        with h5py.File(dataset_path, 'r') as f:
            for i in time_stamp_idx:
                samples_list.append(f[f'samples/t{i}'][:self.num_samples])
                times.append(f.attrs['times'][i])
            true_params = jnp.concatenate((jnp.array([f.attrs['J'], f.attrs['U']]), f.attrs['mu']))

        steps = []
        prev_t = 0.
        for t in times:
            steps.append(int(np.round((t - prev_t) / self.deltat)))
            prev_t = t

        data_indeces = np.arange(self.num_samples)
        return steps, data_indeces, samples_list, true_params
