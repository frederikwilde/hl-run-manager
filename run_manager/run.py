from datetime import datetime
import warnings
import os
from sqlalchemy.types import Integer, Float, String, DateTime, Text
from sqlalchemy.schema import Column
from pathlib import Path
from .series import Series
from . import RESULT_DIR
from . import ORMBase


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
    mps_perturbation = Column(Float)
    opt_method = Column(String(20))
    max_epochs = Column(Integer)
    data_set = Column(String(100))
    time_created = Column(DateTime, nullable=False, default=datetime.utcnow)
    series_name = Column(String(100))
    appendix = Column(Text)

    def save_to_db(self, series: Series):
        if not self.id:
            self.series_name = f'{series.number:03}_{series.name}_{series.hash}'
            series.session.add(self)
            series.session.commit()
        else:
            warnings.warn('Run already saved.')

    def status(self):
        if not self.id:
            return 'NOT_IN_DB'
        warnings.warn('Not implemented yet.')
    
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
        with open(path, 'r') as f:
            out = f.read().split('\n')
        return out

    def result_file_path(self):
        return Path.joinpath(self.output_directory(), Path(f'{self.id}.hdf5'))

    # More auxiliary methods.
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
        return self.__class__(**d)
