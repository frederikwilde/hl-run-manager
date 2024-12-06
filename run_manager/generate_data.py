import os
from typing import List
import jax.numpy as jnp
import jax
from time import time
from dataclasses import dataclass
from datetime import datetime, UTC
import h5py
from pathlib import Path
from differentiable_tebd.physical_models.bose_hubbard_nnn import mps_evolution
from differentiable_tebd.sampling.bosons import sample_from_mps

from run_manager import COMMIT_HASH, DATASET_DIR, load_dir_var
from run_manager.versioning import get_commit_hash
from run_manager.initial_states import ini_mps


@dataclass
class DataSet:
    '''Defines the schema for datasets and provides loading and saving methods `to_hdf5` and `from_hdf5`.'''
    # meta data
    num_sites: int
    true_parameters: jnp.ndarray  # J1, J2, U, *mu
    ini_state: str
    times: List[float]
    # data
    samples_list: List[jnp.ndarray]
    mps_list: List[jnp.ndarray] | None
    errors_squared_list: List[jnp.ndarray]
    # simulation meta data
    chi: int
    deltat: int
    local_dim: int
    prng_seed: int
    run_manager_commit_hash: str
    differentiable_tebd_commit_hash: str
    debug_mode: bool

    def overview(self):
        attrs = [
            'num_sites',
            'true_parameters',
            'ini_state',
            'times',
            'chi',
            'deltat',
            'local_dim',
            'prng_seed',
            'run_manager_commit_hash',
            'differentiable_tebd_commit_hash',
            'debug_mode',
        ]

        lines = []
        for attr in attrs:
            lines.append(attr + ' = ' + f'{getattr(self, attr)}')

        lines.append(f'samples_list: len={len(self.samples_list)} shape={self.samples_list[0].shape}')
        if self.mps_list:
            lines.append(f'mps_list: len={len(self.mps_list)} shape={self.mps_list[0].shape}')
        else:
            lines.append('mps_list: None')
        lines.append(f'errors_squared_list: len={len(self.errors_squared_list)} shape={self.errors_squared_list[0].shape}')

        return '\n'.join(lines)

    def to_hdf5(self, filepath: str | Path):
        with h5py.File(filepath, 'x') as f:
            g_samples = f.create_group('samples')
            g_mps = f.create_group('mps')
            g_errs = f.create_group('errors_squared')

            assert len(self.times) == len(self.samples_list)
            if self.mps_list is None:
                raise ValueError('MPS list not set. This is likely, because the dataset was loaded from another file.')
            assert len(self.times) == len(self.mps_list)
            assert len(self.times) == len(self.errors_squared_list)

            for i, (s, m, e) in enumerate(zip(self.samples_list, self.mps_list, self.errors_squared_list)):
                g_samples.create_dataset(f't{i}', data=s)
                g_mps.create_dataset(f't{i}', data=m)
                g_errs.create_dataset(f't{i}', data=e)

            for k, v in self.__dict__.items():
                if k not in ['samples_list', 'mps_list', 'errors_squared_list']:
                    f.attrs[k] = v

    @classmethod
    def from_hdf5(
        cls,
        filename: str | Path,
        filename_is_path=False,
        *,
        time_stamp_selection: List[int] | None = None,
        load_mps=False,
        num_samples=None,
        parity_project=False
    ):
        if filename_is_path:
            path = filename
        else:
            path = Path.joinpath(Path(DATASET_DIR), Path(filename))

        with h5py.File(path, 'r') as f:
            if time_stamp_selection is None:
                time_stamp_selection = list(range(len(f.attrs['times'])))

            if num_samples is None:
                num_samples = f['samples/t0'].shape[0]

            if load_mps:
                mps_list = [f[f'mps/t{i}'][()] for i in time_stamp_selection]
            else:
                mps_list = None

            samples_list = []
            for i in time_stamp_selection:
                samples = f[f'samples/t{i}'][:num_samples]
                samples_list.append(samples % 2 if parity_project else samples)
            errors_squared_list = [f[f'errors_squared/t{i}'][()] for i in time_stamp_selection]

            meta_data_keys = [
                'num_sites',
                'true_parameters',
                'ini_state',
                'chi',
                'deltat',
                'local_dim',
                'prng_seed',
                'run_manager_commit_hash',
                'differentiable_tebd_commit_hash',
                'debug_mode'
            ]
            kwargs = {k: f.attrs[k] for k in meta_data_keys}

            dataset = cls(
                times=[f.attrs['times'][i] for i in time_stamp_selection],
                samples_list=samples_list,
                mps_list=mps_list,
                errors_squared_list=errors_squared_list,
                **kwargs
            )

        return dataset


def compute_samples(
        name,
        num_sites,
        chi,
        deltat,
        true_parameters,
        local_dim,
        steps,
        seed,
        num_samples,
        ini_state,
        *,
        dataset_containing_mps: str | None = None
    ):
    '''Generate samples.

    Args:
        name (str): File name of the data set.
        num_sites (int)
        chi (int)
        deltat (int)
        true_parameters (jax.numpy.ndarray): Consisting of J, U, *mu
        local_dim (int)
        steps (Sequence[int]): Number of Trotter steps to get to the _next_ time stamp.
            E.g. [10, 10, 10] would generate data after 10, 20, and 30 Trotter steps.
        seed (int): PRNG seed.
        num_samples (int): How many samples to draw per time step.
        ini_state (str): Gets passed on to ini_mps.

    Kwargs:
        dataset_containing_mps (str | None): If set, it must refer to a valid DataSet HDF5 file.
            Uses the MPS from the dataset instead of simulating the time evolution.
    '''
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, (len(steps), num_samples))

    DIFFERENTIABLE_TEBD_DIR = load_dir_var('DIFFERENTIABLE_TEBD_DIR')

    m = ini_mps(num_sites, chi, None, local_dim, ini_state)
    times = []
    samples_list = []
    mps_list = []
    errors_squared_list = []

    if dataset_containing_mps is not None:
        dataset = DataSet.from_hdf5(dataset_containing_mps, num_samples=1, load_mps=True)
        mps_iterator = iter(dataset.mps_list)
        errors_iterator = iter(dataset.errors_squared_list)

        # Some checks to ensure that the the dataset matches the parameters
        # specified in the arguments.
        _times = []
        for s in steps:
            t = s * deltat + _times[-1] if _times else s * deltat
            _times.append(t)

        assert num_sites == dataset.num_sites
        assert chi == dataset.chi
        assert deltat == dataset.deltat
        assert jnp.allclose(true_parameters, dataset.true_parameters)
        assert local_dim == dataset.local_dim
        assert all(t1 == t2 for t1, t2 in zip(_times, dataset.times))
        assert ini_state == dataset.ini_state

    for s, k in zip(steps, keys):
        time_stamp = deltat * s + times[-1] if times else deltat * s
        times.append(time_stamp)

        if dataset_containing_mps is not None:
            m, errors_squared = next(mps_iterator), next(errors_iterator)
        else:
            t = time()
            m, errors_squared = mps_evolution(true_parameters, deltat, s, m)
            print(f'Time evolution finished in {time() - t:.3f}s')

        t = time()
        samples = sample_from_mps(m, k)
        print(f'Sampling finished in {time() - t:.3f}s')

        samples_list.append(samples)
        mps_list.append(m)
        errors_squared_list.append(errors_squared)

    dataset = DataSet(
        num_sites=num_sites,
        true_parameters=true_parameters,
        ini_state=ini_state,
        times=times,
        samples_list=samples_list,
        mps_list=mps_list,
        errors_squared_list=errors_squared_list,
        chi=chi,
        deltat=deltat,
        local_dim=local_dim,
        prng_seed=seed,
        run_manager_commit_hash=COMMIT_HASH,
        differentiable_tebd_commit_hash=get_commit_hash(DIFFERENTIABLE_TEBD_DIR),
        debug_mode=os.environ.get('DEBUG') == '1'
    )

    now = datetime.now(UTC)
    filename = f'{now:%y-%m-%d}-{name}-{ini_state}-n{num_sites}.hdf5'
    path = Path.joinpath(Path(DATASET_DIR), Path(filename))

    dataset.to_hdf5(path)
