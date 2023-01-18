'''Main script to be run for one optimization process.'''

import logging
import jax.numpy as jnp
import numpy as np
import sys
import jax
from pathlib import Path
from tqdm import tqdm
from time import time
import h5py
from differentiable_tebd.utils.mps import mps_zero_state, norm_squared
from differentiable_tebd.utils.mps_bosons import probability
from differentiable_tebd.physical_models.bose_hubbard import mps_evolution_order2
from differentiable_tebd.sampling.bosons import one_sample_from_mps
from adam import Adam

from . import Series
from . import Run
from . import DATASET_DIR


MAIN_PATH = Path(__file__)


def ini_mps(n, chi, pert, local_dim, occupation):
    m = mps_zero_state(n, chi, pert, d=local_dim)
    if occupation == 'neel':
        for i in range(n//2):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)
    elif occupation == 'half-filled':
        for i in range(0, n, 2):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)
    else:
        raise ValueError('Invalid occupation.')
    return m


def negative_log_likelihood(sample, mps):
    return - jnp.log(probability(mps, sample))
# vectorize over bitstrings
batched_nll = jax.jit(jax.vmap(negative_log_likelihood, in_axes=(0, None)))
def loss(params, mps, deltat, steps, samples_list, total_num_samples):
    nll = 0.
    for i in jnp.arange(len(steps), dtype=int):
        mps, _ = mps_evolution_order2(params, deltat, steps[i], mps)
        nll += jnp.sum(batched_nll(samples_list[i], mps))
    return nll / total_num_samples


def load_data(time_stamp_idx, run):
    dataset_path = Path.joinpath(Path(DATASET_DIR), Path(run.data_set))
    ini_points = f'initial_points_{run.num_sites}.hdf5'
    initial_point_path = Path.joinpath(Path(DATASET_DIR), Path(ini_points))
    samples_list = []
    for s in cumulated_steps[:num_timestamps]:
        with h5py.File(f'samples{s}.hdf5', 'r') as f:
            samples_list.append(f['samples'][()])
    data_indeces = np.arange(len(samples_list[0]))
    return steps, initial_params, data_indeces


def execute(series_number, run_index):
    series = Series(series_number)
    run: Run = series.session.query(Run).where(Run.id == run_index).first()
    run.pre_execute_check()

    n = run.num_sites
    chi = run.chi
    deltat = run.deltat
    d = run.local_dim
    time_stamp_idx = [int(i) for i in run.time_stamps.split(',')]

    steps, params, data_indeces = load_data(time_stamp_idx, run)

    opt = Adam(params)
    opt.step_size = run.step_size
    loss_history, param_history, grad_history = [], [], []
    for e in range(run.max_epochs):
        rng.shuffle(data_indeces)
        for batch_indeces in data_indeces.reshape(len(data_indeces)//run.batch_size, run.batch_size):
            v, g = jax.value_and_grad(loss)(
                opt.parameters,
                ini_mps(n, chi, 1e-6, d),
                deltat,
                steps[:num_timestamps],
                [s[batch_indeces] for s in samples_list],
                num_timestamps * batch_size
            )
            loss_history.append(v)
            param_history.append(opt.parameters)
            grad_history.append(g)
            opt.step(g)
            print(f'parameters {opt.parameters}')
        print(f'Epoch {e+1} done\n')


    with h5py.File('half-history-ts2-stp05.hdf5', 'x') as f:
        f.create_dataset('loss_history', data=loss_history)
        f.create_dataset('param_history', data=param_history)
        f.create_dataset('grad_history', data=grad_history)
        f.attrs['chi'] = chi
        f.attrs['d'] = d
        f.attrs['num_timestamps'] = num_timestamps
        f.attrs['batch_size'] = batch_size


if __name__ == '__main__':
    series_number = int(sys.argv[1])
    run_index = int(sys.argv[2])

    execute(series_number, run_index)
