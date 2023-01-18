import sys
import jax.numpy as jnp
import jax
from tqdm import tqdm
import h5py
from differentiable_tebd.utils.mps import mps_zero_state, norm_squared
from differentiable_tebd.utils.mps_bosons import probability
from differentiable_tebd.physical_models.bose_hubbard import mps_evolution_order2
from differentiable_tebd.sampling.bosons import one_sample_from_mps

from . import COMMIT_HASH
from .main import ini_mps


steps = int(sys.argv[1])
key = jax.random.PRNGKey(steps)
keys = jax.random.split(key, 1000)

def compute_samples(steps, keys):
    n, chi, deltat = 50, 30, .02
    t, u, mu = 1., .1, .5
    params = jnp.array([t, u, mu])

    m = ini_mps(n, chi, None, 5, 'neel')
    
    m, errors_squared = mps_evolution_order2(params, deltat, steps, m)

    samples = []
    for i, k in enumerate(keys):
        if i % 10 == 0:
            print(f'Computing {i}-th sample.')
        samples.append(one_sample_from_mps(m, k))

    with h5py.File(f'samples-neel{steps}.hdf5', 'x') as f:
        f.create_dataset('samples', data=samples)
        f.create_dataset('mps', data=m)
        f.create_dataset('errors_squared', data=errors_squared)
        f.attrs['t'] = t
        f.attrs['u'] = u
        f.attrs['mu'] = mu
        f.attrs['deltat'] = deltat
        f.attrs['run_manager_commit_hash'] = COMMIT_HASH


if __name__ == '__main__':
    compute_samples(steps, keys)
