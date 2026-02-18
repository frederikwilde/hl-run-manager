import os
import shutil
import logging
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import pytest

os.environ['DEBUG'] = '1'
os.environ['SVD_GRAD_THRESHOLD'] = os.environ.get('SVD_GRAD_THRESHOLD', '1e-8')

from differentiable_tebd.utils.mps_bosons import probability, parity_projection
from differentiable_tebd.physical_models.bose_hubbard_nnn import mps_evolution

from run_manager import Series, Run
from run_manager.adam import Optimizer


@pytest.mark.parametrize('pproj', [True, False])
def test_pipeline(pproj):
    # Create series and a run object
    series = Series.new('test')
    # NOTE: When these datasets get archived in the future this test must be updated.
    data_sets = (
        '25-01-14-samp10000-seed10000-t10-neel-n12.hdf5, '
        '25-01-14-samp10000-seed10000-t10-neel-reverse-n12.hdf5'
    )
    run = Run(
        initial_point_seed = 42,
        initial_point_bias = 0.01,
        initial_point_stddev = 0.05,
        step_size = 2e-3,
        deltat = 0.5,
        time_stamps = '4,5',
        chi = 20,
        local_dim = 3,
        num_samples = 10000,
        batch_size = 10000,
        mps_perturbation = 0,
        max_epochs = 20,
        data_sets = data_sets,
        parity_project = pproj,
        bfgs_gtol = 1e-3,
        bfgs_maxiter = 100,
        appendix = ''
    )
    run.add_to_db(series)

    # Define loss and execute run
    def negative_log_likelihood(sample, mps):
        return - jnp.log(probability(mps, sample))

    # vectorize over bitstrings
    batched_nll = jax.jit(jax.vmap(negative_log_likelihood, in_axes=(0, None)))

    def loss(params, mps, deltat, steps, samples_list, total_num_samples, parity_project):
        nll = 0.
        for nsteps, samples in zip(steps, samples_list):
            mps, _ = mps_evolution(params, deltat, nsteps, mps, checkpoint=False)
            m = parity_projection(mps) if parity_project else mps
            nll += jnp.sum(batched_nll(samples, m))

        # regularization = jnp.sum(10 * (params[:3] - jnp.array([.2, .01, 1.])) ** 2)
        return nll / total_num_samples  # + regularization

    def initialization(run: Run):
        # perturbation around true params since we assume we know them approximately
        true_params = run.get_true_params_from_dataset()

        key = jax.random.PRNGKey(run.initial_point_seed)
        std = run.initial_point_stddev
        bias = run.initial_point_bias
        noise = std * (jax.random.normal(key, true_params.shape) + bias)

        # noise = noise.at[:3].set(0)
        return noise + true_params

    launch_script = 'launch script placeholder'
    slurm_job_id = 'slurm job id placeholder'

    run.execute(loss, initialization, Optimizer, launch_script, slurm_job_id, print_progress=True)

    # Load results and verify
    with h5py.File(run.result_file_path(), 'r') as f:
        param_history = f['param_history'][()]

    J1, J2, U, *mu = run.get_true_params_from_dataset()

    def errors(params):
        J1_est, J2_est, U_est, *mu_est = params

        J1_err = np.abs(J1 - J1_est)
        J2_err = np.abs(J2 - J2_est)
        U_err = np.abs(U - U_est)
        mu_dists = np.convolve(mu, [1, -1], mode='valid')
        mu_est_dists = np.convolve(mu_est, [1, -1], mode='valid')
        mu_err = np.linalg.norm(mu_dists - mu_est_dists) / mu_dists.size
        return J1_err, J2_err, U_err, mu_err

    print(errors(param_history[-1]))

    # Delete Series and results
    # Somehow deleting the test series within the interpreter session still
    # does not work. Need to delete manually afterwards.

    # logging.shutdown()
    # series.session.close()
    # series.session.connection().engine.dispose()
    # shutil.rmtree(series.path)


if __name__ == "__main__":
    test_pipeline(False)
