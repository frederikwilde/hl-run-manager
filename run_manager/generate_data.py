import sys
import jax.numpy as jnp
import jax
from time import time
from datetime import datetime
import h5py
from pathlib import Path
from differentiable_tebd.physical_models.bose_hubbard import mps_evolution_order2
from differentiable_tebd.sampling.bosons import sample_from_mps
from differentiable_tebd.utils.mps import mps_zero_state

from . import COMMIT_HASH
from . import DATASET_DIR
from .versioning import get_differentiable_tebd_commit_hash


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
        params = jnp.zeros(2 + len(m), dtype=jnp.float64).at[0].set(1.)
        m, _ = mps_evolution_order2(params, T/10, 10, m)

    else:
        raise ValueError('Invalid occupation.')

    return m


def compute_samples(name, cf, steps, num_keys, ini_state_occupation):
    '''Generate samples.

    Args:
        name (str): File name of the data set.
        cf (object): SimpleNamespace object which has the following attributes.
            num_sites, chi, deltat, J, U, mu, local_dim
        steps (Sequence[int]): Number of Trotter steps to get to the _next_ time stamp.
            E.g. [10, 10, 10] would generate data after 10, 20, and 30 Trotter steps.
        num_keys (int): How many samples to draw.
        ini_state_occupation (str): Gets passed on to ini_mps.
    '''
    key = jax.random.PRNGKey(steps[0])
    keys = jax.random.split(key, num_keys)

    cf.run_manager_commit_hash = COMMIT_HASH
    cf.differentiable_tebd_hash = get_differentiable_tebd_commit_hash()

    params = jnp.concatenate([jnp.array([cf.J, cf.U]), cf.mu])

    m = ini_mps(cf.num_sites, cf.chi, None, cf.local_dim, ini_state_occupation)
    cf.times = []

    now = datetime.utcnow()
    filename = f'{now:%y-%m-%d}-{name}-{ini_state_occupation}-n{cf.num_sites}.hdf5'
    path = Path.joinpath(Path(DATASET_DIR), Path(filename))
    with h5py.File(path, 'x') as f:
        g_samples = f.create_group('samples')
        g_mps = f.create_group('mps')
        g_errs = f.create_group('errors_squared')

        for i, s in enumerate(steps):
            time_stamp = cf.deltat * s + cf.times[-1] if cf.times else cf.deltat * s
            cf.times.append(time_stamp)

            t1 = time()
            m, errors_squared = mps_evolution_order2(params, cf.deltat, s, m)
            t2 = time()
            evol_time = t2 - t1
            print(f'Time evolution finished in {evol_time:.3f}s')
            samples = sample_from_mps(m, keys)
            sample_time = time() - t2
            print(f'Sampling finished in {sample_time:.3f}s')

            g_samples.create_dataset(f't{i}', data=samples)
            g_mps.create_dataset(f't{i}', data=m)
            g_errs.create_dataset(f't{i}', data=errors_squared)
        
        for k, v in cf.__dict__.items():
            f.attrs[k] = v
