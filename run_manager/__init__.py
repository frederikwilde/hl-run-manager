import os
from configparser import ConfigParser
from sqlalchemy.orm import declarative_base
from .versioning import get_commit_hash


config = ConfigParser()
config.read('config.ini')


def load_dir_var(var: str):
    try:
        out = os.environ[var]
    except KeyError:
        out = config['Environment'][var]

    if not os.path.exists(out):
        raise EnvironmentError(f'{var} is {out}, which is not a valid path.')
    return out


ORMBase = declarative_base()

RESULT_DIR = load_dir_var('RESULT_DIR')
DATASET_DIR = load_dir_var('DATASET_DIR')
RUN_MANAGER_DIR = load_dir_var('RUN_MANAGER_DIR')

COMMIT_HASH = get_commit_hash(RUN_MANAGER_DIR)
'''First 6 symbols of the hash of the currently checked out commit of the run_manager repository.'''


from .run import Run
from .series import Series
