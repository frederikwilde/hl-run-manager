import sys
import jax.numpy as jnp
import jax
from time import time
from datetime import datetime
import h5py
from types import SimpleNamespace
from pathlib import Path
from differentiable_tebd.physical_models.bose_hubbard import mps_evolution_order2
from differentiable_tebd.sampling.bosons import sample_from_mps

from . import COMMIT_HASH
from . import DATASET_DIR
from .main import ini_mps
from .versioning import get_differentiable_tebd_commit_hash


def compute_samples(steps, num_keys):
    key = jax.random.PRNGKey(steps)
    keys = jax.random.split(key, num_keys)
    cf = SimpleNamespace()  # config object

    cf.run_manager_commit_hash = COMMIT_HASH
    cf.differentiable_tebd_hash = get_differentiable_tebd_commit_hash()

    cf.num_sites = 50
    cf.chi = 30
    cf.deltat = .02
    cf.steps = steps

    cf.J = .1
    cf.U = 1.
    cf.mu = .5 * (1. - jnp.sin(jnp.linspace(0, jnp.pi, cf.num_sites)))
    params = jnp.concatenate([jnp.array([cf.J, cf.U]), cf.mu])

    m = ini_mps(cf.num_sites, cf.chi, None, 4, 'neel')

    t1 = time()
    m, errors_squared = mps_evolution_order2(params, cf.deltat, steps, m)
    t2 = time()

    cf.evol_time = t2 - t1
    print(f'Time evolution finished in {cf.evol_time:.3f}s')

    samples = sample_from_mps(m, keys)
    cf.sample_time = time() - t2
    print(f'Sampling finished in {cf.sample_time:.3f}s')

    now = datetime.utcnow()
    filename = f'{now:%y-%m-%d}-bowl-neel-n{cf.num_sites}-steps{steps}.hdf5'
    path = Path.joinpath(Path(DATASET_DIR), Path(filename))
    with h5py.File(path, 'x') as f:
        f.create_dataset('samples', data=samples)
        f.create_dataset('mps', data=m)
        f.create_dataset('errors_squared', data=errors_squared)
        for k, v in cf.__dict__.items():
            f.attrs[k] = v


if __name__ == '__main__':
    steps = int(sys.argv[1])
    num_keys = int(sys.argv[2])

    compute_samples(steps, num_keys)
