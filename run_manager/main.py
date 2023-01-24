'''Main script to be run for one optimization process.'''

import warnings
import logging
import jax.numpy as jnp
import numpy as np
import sys
import jax
from pathlib import Path
from tqdm import tqdm
from time import time
import h5py
from typing import Union
from differentiable_tebd.utils.mps import mps_zero_state, norm_squared
from differentiable_tebd.utils.mps_bosons import probability
from differentiable_tebd.physical_models.bose_hubbard import mps_evolution_order2
from differentiable_tebd.sampling.bosons import sample_from_mps


from .adam import Adam
from .series import Series
from .run import Run
from . import DATASET_DIR

MAIN_PATH = Path(__file__)


def ini_mps(n, chi, pert, local_dim, occupation, rng=None):
    m = mps_zero_state(n, chi, pert, d=local_dim, rng=rng)
    if occupation == 'half-filled':
        for i in range(n//2):
            m = m.at[i, 0, 0, 0].set(0.).at[i, 0, 1, 0].set(1.)
    elif occupation == 'neel':
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
    
    mu = params[2:]
    regularization = .1 * jnp.sum((mu[:-1] - mu[1:]) ** 2)
    return nll / total_num_samples + regularization


def load_data(time_stamp_idx, run):
    dataset_path = Path.joinpath(Path(DATASET_DIR), Path(run.data_set))

    samples_list, times = [], []
    with h5py.File(dataset_path, 'r') as f:
        for i in time_stamp_idx:
            samples_list.append(f[f'samples/t{i}'][:run.num_samples])
            times.append(f.attrs['times'][i])
        true_params = jnp.concatenate((jnp.array([f.attrs['J'], f.attrs['U']]), f.attrs['mu']))

    steps = []
    prev_t = 0.
    for t in times:
        steps.append(int(np.round((t - prev_t) / run.deltat)))
        prev_t = t

    data_indeces = np.arange(run.num_samples)
    return steps, data_indeces, samples_list, true_params


def initial_parameters(run: Run):
    key = jax.random.PRNGKey(run.initial_point_seed)
    return jax.random.uniform(key, (run.num_sites + 2,))


def execute(
        series_number: Union[int, None] = None,
        run_index: Union[int, None] = None,
        run: Union[Run, None] = None
    ):
    '''
    Execute the optimization process and store the results.
    As input either a `Run` object can be provided or a series number and the corresponding run index.
    '''
    if not run:
        series = Series(series_number)
        run = series.session.query(Run).where(Run.id == run_index).first()
    elif (series_number is not None) or (run_index is not None):
        warnings.warn('A run was explicitly provided and the arguments series_number and run_index will be ignored.')
    run.pre_execute_check()

    n = run.num_sites
    chi = run.chi
    deltat = run.deltat
    d = run.local_dim
    time_stamp_idx = [int(i) for i in run.time_stamps.split(',')]

    steps, data_indeces, samples_list, true_params = load_data(time_stamp_idx, run)
    params = .01 * initial_parameters(run)
    # params = params.at[1].add(1.)

    opt = Adam(params)
    opt.step_size = run.step_size
    loss_history, param_history, grad_history = [], [], []
    data_shuffle_rng_seed = run.id
    rng = np.random.default_rng(data_shuffle_rng_seed)

    filename = Path.joinpath(run.output_directory(), Path(f'{run.id}.hdf5'))
    
    for e in range(run.max_epochs):
        rng.shuffle(data_indeces)
        for batch_indeces in data_indeces.reshape(len(data_indeces)//run.batch_size, run.batch_size):
            t1 = time()
            v, g = jax.value_and_grad(loss)(
                opt.parameters,
                ini_mps(n, chi, run.mps_perturbation, d, 'neel', rng=rng),
                deltat,
                steps,
                [s[batch_indeces] for s in samples_list],
                len(samples_list) * run.batch_size
            )
            loss_history.append(v)
            param_history.append(opt.parameters)
            grad_history.append(g)
            opt.step(g)

            diffs = opt.parameters - true_params
            J_error = np.abs(diffs[0])
            U_error = np.abs(diffs[1])
            mu_avg_error = np.linalg.norm(diffs[2:]) / n
            print((
                f'Time: {time()-t1:.2f}s  '
                f'Errors J: {J_error:.05f} U: {U_error:.05f} mu: {mu_avg_error:.05f}'
            ))

        with h5py.File(filename, 'w') as f:
            f.create_dataset('loss_history', data=loss_history)
            f.create_dataset('param_history', data=param_history)
            f.create_dataset('grad_history', data=grad_history)
            f.attrs['data_shuffle_rng_seed'] = data_shuffle_rng_seed
            f.attrs['steps'] = steps
            f.attrs['true_params'] = true_params
        print(f'Epoch {e+1} done\n')


if __name__ == '__main__':
    series_number = int(sys.argv[1])
    run_index = int(sys.argv[2])

    execute(series_number, run_index)
