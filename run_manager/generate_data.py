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
    elif occupation == 'unity':
        for i in range(0, num_sites):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)
    else:
        raise ValueError('Invalid occupation.')
    return m


def compute_samples(name, cf, steps, num_keys, ini_state_occupation):
    '''Generate samples.

    Args:
        name (str): File name of the data set.
        cf (object): SimpleNamespace object which has the following attributes.
            num_sites, chi, deltat, J, U, mu, local_dim
        steps (int): Number of time steps.
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
