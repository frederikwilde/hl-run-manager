
import os
from datetime import datetime
import numpy as np
import sqlalchemy as sqa
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy.sql.expression import text
from sqlalchemy.types import Integer, Float, String, DateTime, Text
from sqlalchemy.schema import Column
import time


ORMBase = declarative_base()


class Run(ORMBase):
    '''Each instance needs a time_selection attribute as an iterable of integers
    before saving it to the database.'''
    __tablename__ = 'Runs'
    id = Column(Integer, primary_key=True)
    initial_point_index = Column(Integer)
    num_sites = Column(Integer)
    step_size = Column(Float)
    deltat = Column(Float)
    time_stamps = Column(String(50))
    chi = Column(Integer)
    num_bases = Column(Integer)
    num_samples_per_basis = Column(Integer)
    batch_size = Column(Integer)
    mps_perturbation = Column(Float)
    gtol = Column(Float)
    opt_method = Column(String(20))
    full_gradient_batch_threshold = Column(Integer)
    time_created = Column(DateTime, nullable=False, default=datetime.utcnow)
    progress = Column(Text, default='Run object created')
    appendix = Column(Text, default='')
    
    def __repr__(self):
        if self.time_created is None:
            return '<Run object>'
        else:
            return f'{self.id}-{self.time_created:%y%m%d-%H%M%S}'

    def save_to_db(self):
        self.time_stamps = ','.join([str(t) for t in self.time_selection])
        return self._commit()
    
    def compute_batch_sizes(self):
        r = self.num_bases * self.num_samples_per_basis / self.full_gradient_batch_threshold
        if r <= 1:
            self.full_gradient_batch_size = None
        else:
            self.full_gradient_batch_size = int(self.num_samples_per_basis / r)
        self.batch_size = int(max([1, 100 / self.num_bases]))
    
    def add_attributes_to_hdf5(self, file):
        for k, v in self.__dict__.items():
            try:
                file.attrs[k] = v
            except TypeError:
                file.attrs[k] = str(v)
    
    def copy(self):
        d = dict(self.__dict__)
        d.pop("_sa_instance_state")
        return self.__class__(**d)

