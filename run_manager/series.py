import sqlalchemy as sqa
from sqlalchemy.orm import Session
import os
from pathlib import Path

from run import ORMBase
from . import COMMIT_HASH


def all_series(hash_only=False):
    pass


def highest_series_number(result_dir):
    pass


class Series:
    def __init__(self, name):
        '''Set up a new series directory.'''
        number = highest_series_number()
        full_name = f'{number}_{name}_{COMMIT_HASH[-6:]}'
        pass

    def load(self, number=None, name=None):
        '''Load and existing series.'''
        if number is None and name is None:
            raise ValueError('Either name or number needs to be specify to load a series.')
    
    def _load_db_session(self):
        self.db_path = f'{}/results.db'
        self.engine = sqa.create_engine(f'sqlite:///{self.db_path}')
        self.session = Session(self.engine)
    
    def _create_database(self):
        ORMBase.metadata.create_all(self.engine)