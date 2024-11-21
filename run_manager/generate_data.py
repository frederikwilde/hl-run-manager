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
from differentiable_tebd.utils.mps import mps_zero_state

from run_manager import COMMIT_HASH, DATASET_DIR, load_dir_var
from run_manager.versioning import get_commit_hash


def _to_neel(mps, reverse=False):
    '''Convert mps_zero_state into Neel state.
    By default 101010... If reverse is False the order is 010101...'''
    start = 1 if reverse else 0
    for i in range(start, len(mps), 2):
        mps = mps.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)
    return mps


def _0_2_quench(m, T, reverse=False):
    '''Quench with J1=0.2, J2=0.01, U=1. for time T.'''
    T = jnp.pi / 8
    m = _to_neel(m, reverse)
    params = jnp.array([0.2, 0.01, 1.] + len(m) * [0.])
    m, _ = mps_evolution(params, T/10, 10, m)
    return m


def ini_mps(num_sites, chi, mps_perturbation, local_dim, occupation, rng=None):
    '''Creates specific MPS, such as the Neel state and others.

    Args:
        num_sites (int)
        chi (int)
        mps_perturbation (float)
        local_dim (int)
        occupation (str): The following options are valid:
            'half-filled': Only the left half of the system is filled,
                each site with one particle.
            'neel': Every other site is filled, beginning with a filled
                site. 101010...
            '2-3rds-neel': 011011...
            '1-3rd-neel': 0100100...
            'dimer': A Neel state which has evolved up to time pi/4 under
                nearest-neighbor hopping between disjunct pairs of sites
                (0-1, 2-3, 4-5, ... but not 1-2, 3-4, ...)
                One dimer is given by: 1/sqrt(2) * (|10> - i|01>)
            'n-mer': A Neel state which has evolved up to time pi/8 under
                all nearest-neighor hopping terms. Note that this requires a
                local dimension of at least 5.
            'n-mer-interacting': A Neel state which has evolved up to time pi/8 under
                all nearest-neighor hopping terms and on-site interactions.
            'n-mer-pi_4': Like n-mer, but evolved to time pi/4

            '0_2-quench-{rev}-{T}': A quench of the Neel state with the following
                parameters: J1 = 0.2, J2 = 0.01, U = 1. The Neel state is 101010...
                and if 'rev' is specified, it is reversed, i.e., 010101...
    '''
    m = mps_zero_state(
        num_sites,
        chi,
        mps_perturbation,
        d=local_dim,
        rng=rng
    )
    if occupation == 'half-filled':
        for i in range(num_sites//2):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)

    elif occupation == 'neel':
        for i in range(0, num_sites, 2):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)

    elif occupation == '2-3rds-neel':
        for i in range(0, num_sites):
            if i % 3 == 1 or i % 3 == 2:
                m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)

    elif occupation == '1-3rd-neel':
        for i in range(0, num_sites):
            if i % 3 == 1:
                m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)

    elif occupation == 'unity':
        for i in range(0, num_sites):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)

    elif occupation == 'dimer':
        s = .5 ** (1/4)
        for i in range(0, num_sites, 2):
            m = m.at[i, 0, 0, 0].set(1j * s)
            m = m.at[i, 0, 1, 1].set(-s)
            m = m.at[i+1, 0, 0, 0].set(0)
            m = m.at[i+1, 0, 1, 0].set(-s)
            m = m.at[i+1, 1, 0, 0].set(-s)

    elif occupation == 'semi-dimer':
        raise NotImplementedError('To do!')

    elif occupation == 'n-mer':
        T = jnp.pi / 8
        # initialize Neel state
        for i in range(0, num_sites, 2):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)
        params = jnp.zeros(3 + len(m), dtype=jnp.float64).at[0].set(1.)
        m, _ = mps_evolution(params, T/10, 10, m)

    elif occupation == 'n-mer-interacting':
        T = jnp.pi / 8
        # initialize Neel state
        for i in range(0, num_sites, 2):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)
        params = jnp.zeros(3 + len(m), dtype=jnp.float64).at[0].set(1.)
        params = params.at[2].set(1.)
        m, _ = mps_evolution(params, T/10, 10, m)

    elif occupation == 'n-mer-pi_4':
        T = jnp.pi / 4
        # initialize Neel state
        for i in range(0, num_sites, 2):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)
        params = jnp.zeros(3 + len(m), dtype=jnp.float64).at[0].set(1.)
        m, _ = mps_evolution(params, T/10, 10, m)

    #### QUENCHES WITH J/U = 0.2 and next-nearest-neighbor hopping

    elif occupation == '0_2-quench-pi_8':
        m = _0_2_quench(m, jnp.pi / 8)

    elif occupation == '0_2-quench-rev-pi_8':
        m = _0_2_quench(m, jnp.pi / 8, reverse=True)

    elif occupation == '0_2-quench-pi_4':
        m = _0_2_quench(m, jnp.pi / 4)

    elif occupation == '0_2-quench-rev-pi_4':
        m = _0_2_quench(m, jnp.pi / 4, reverse=True)

    elif occupation == '0_2-quench-3pi_8':
        m = _0_2_quench(m, 3 * jnp.pi / 8)

    elif occupation == '0_2-quench-rev-3pi_8':
        m = _0_2_quench(m, 3 * jnp.pi / 8, reverse=True)

    else:
        raise ValueError('Invalid occupation.')

    return m


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
                if not k in ['samples_list', 'mps_list', 'errors_squared_list']:
                    f.attrs[k] = v

    @classmethod
    def from_hdf5(
        cls,
        filename: str | Path,
        filename_is_path=False,
        *,
        time_stamp_selection: List[int] | None = None,
        load_mps=False,
        num_samples=None
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

            samples_list = [f[f'samples/t{i}'][:num_samples] for i in time_stamp_selection]
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