import sqlalchemy as sqa
from sqlalchemy.orm import Session
import os
from pathlib import Path
from datetime import datetime, UTC
import warnings
import numpy as np

from run_manager import ORMBase, COMMIT_HASH, RESULT_DIR


def existing_series():
    series = [f for f in os.scandir(RESULT_DIR) if f.is_dir()]
    series_numbers = np.array([int(s.name[:3]) for s in series])
    sorted_indices = np.argsort(series_numbers)
    return [series[i] for i in sorted_indices]


def max_series_number():
    series = existing_series()
    if series:
        return int(series[-1].name[:3])
    else:
        return 0


class Series:
    '''Load an existing series.'''
    def __init__(self, number):
        for s in existing_series():
            num, name, hash = int(s.name[:3]), s.name[4:-7], s.name[-6:]
            if number == num:
                break
        else:
            raise ValueError(f'No series with number {number} found.')

        if not (hash == COMMIT_HASH):
            msg = (
                f'The series {num}_{name} refers to commit {hash}, '
                f'but the currently checked out commit is {COMMIT_HASH}. '
                f'Checkout {hash} and reload the series.'
            )
            if os.environ.get('DEBUG') == '1':
                warnings.warn(msg + '\n\nDEBUG mode activated. The database might not match the ORM model.')
            else:
                # TODO: Should not raise a ValueError but some configuration error.
                raise ValueError(msg)

        self.path = s.path
        self.number = num
        self.name = name
        self.hash = hash
        self._load_db_session()

    def _load_db_session(self):
        db_path = Path.joinpath(Path(self.path), Path('results.db'))
        self.engine = sqa.create_engine(f'sqlite:///{db_path}')
        self.session = Session(self.engine)

    def __repr__(self):
        return f'<Series {self.number}: {self.name}>'

    @classmethod
    def new(cls, name):
        '''Set up a new series.'''
        if os.environ.get('DEBUG') == '1':
            warnings.warn('Creating series in debug mode!')
            name = name + '_DEBUGMODE'

        number = max_series_number() + 1
        full_name = f'{number:03}_{name}_{COMMIT_HASH}'
        path = Path.joinpath(Path(RESULT_DIR), Path(full_name))

        os.mkdir(path)
        os.mkdir(Path.joinpath(path, Path('output')))
        os.mkdir(Path.joinpath(path, Path('scripts')))

        with open(Path.joinpath(path, Path('readme.txt')), 'x') as f:
            f.write(f'{name}\n{datetime.now(UTC):%y-%m-%d %H:%M:%S}\n\n')

        db_path = Path.joinpath(path, Path('results.db'))
        engine = sqa.create_engine(f'sqlite:///{db_path}')
        ORMBase.metadata.create_all(engine)

        return cls(number)

    def delete_run(self, run):
        response = input(f"{run} will be removed from the database. Continue? [y/n]\n")

        if response == 'y':
            self.session.delete(run)
            self.session.commit()
            print("Deleted.")
        else:
            print('Aborted.')
