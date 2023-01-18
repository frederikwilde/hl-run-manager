import os
import json
from sqlalchemy.orm import declarative_base
from pathlib import Path
from .versioning import get_commit_hash


try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    pass

def load_dir_var(var: str):
    try:
        out = os.environ[var]
    except KeyError:
        out = config[var]
    if not os.path.exists(out):
        raise EnvironmentError(f'{var} is {out}, which is not a valid path.')
    return out


ORMBase = declarative_base()
RESULT_DIR = load_dir_var('RESULT_DIR')
DATASET_DIR = load_dir_var('DATASET_DIR')
PACKAGE_PATH = Path(__file__).parent.parent
COMMIT_HASH = get_commit_hash()
'''Last 6 symbols of the hash of the currently checked out commit of the run_manager repository.'''


from .run import Run
from .series import Series
