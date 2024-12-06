import numpy as np
import jax.numpy as jnp
import pytest

from run_manager.initial_states import ini_mps
from differentiable_tebd.physical_models.bose_hubbard_nnn import mps_evolution


@pytest.mark.parametrize('n', [3, 4, 5])
@pytest.mark.parametrize('chi', [1, 4])
@pytest.mark.parametrize('d', [2, 3])
def test_neel_state(n, chi, d):
    m = ini_mps(n, chi, None, d, 'neel')
    m_rev = ini_mps(n, chi, None, d, 'neel-reverse')

    for i in range(n):
        if i % 2 == 0:
            m = m.at[i, 0, 1, 0].add(-1)
            m_rev = m_rev.at[i, 0, 0, 0].add(-1)
        else:
            m = m.at[i, 0, 0, 0].add(-1)
            m_rev = m_rev.at[i, 0, 1, 0].add(-1)

    assert jnp.allclose(m, 0)
    assert jnp.allclose(m_rev, 0)


@pytest.mark.parametrize('n', [3, 6])
def test_half_filled(n):
    m = ini_mps(n, 2, None, 2, 'half-filled')
    m_rev = ini_mps(n, 2, None, 2, 'half-filled-reverse')

    m = m.at[:n//2, 0, 1, 0].add(-1)
    m = m.at[n//2:, 0, 0, 0].add(-1)
    m_rev = m_rev.at[n//2:, 0, 1, 0].add(-1)
    m_rev = m_rev.at[:n//2, 0, 0, 0].add(-1)

    assert jnp.allclose(m, 0)
    assert jnp.allclose(m_rev, 0)


@pytest.mark.parametrize('n', [6, 7, 8])
def test_neel_one_third(n):
    m = ini_mps(n, 2, None, 2, 'neel-one-third')
    m0 = ini_mps(n, 2, None, 2, 'neel-one-third-start0')
    m1 = ini_mps(n, 2, None, 2, 'neel-one-third-start1')
    m2 = ini_mps(n, 2, None, 2, 'neel-one-third-start2')

    assert jnp.allclose(m, m0)

    for i in range(0, n, 3):
        m = m.at[i, 0, 1, 0].add(-1)
        m1 = m1.at[i, 0, 0, 0].add(-1)
        m2 = m2.at[i, 0, 0, 0].add(-1)

    for i in range(1, n, 3):
        m = m.at[i, 0, 0, 0].add(-1)
        m1 = m1.at[i, 0, 1, 0].add(-1)
        m2 = m2.at[i, 0, 0, 0].add(-1)

    for i in range(2, n, 3):
        m = m.at[i, 0, 0, 0].add(-1)
        m1 = m1.at[i, 0, 0, 0].add(-1)
        m2 = m2.at[i, 0, 1, 0].add(-1)

    assert jnp.allclose(m, 0)
    assert jnp.allclose(m1, 0)
    assert jnp.allclose(m2, 0)


@pytest.mark.parametrize('time', [.5, 1.])
def test_quench(time):
    m = ini_mps(6, 15, None, 3, f'quench-time{time}')
    m_rev = ini_mps(6, 15, None, 3, f'quench-time{time}-reverse')

    m_expected = ini_mps(6, 15, None, 3, 'neel')
    m_rev_expected = ini_mps(6, 15, None, 3, 'neel-reverse')
    params = jnp.array([0.2, 0.01, 1.] + 6 * [0.])
    m_expected, _ = mps_evolution(params, time/100, 100, m_expected)
    m_rev_expected, _ = mps_evolution(params, time/100, 100, m_rev_expected)

    assert jnp.allclose(m, m_expected)
    assert jnp.allclose(m_rev, m_rev_expected)


def test_perturbation():
    rng = np.random.default_rng(42)
    m1 = ini_mps(6, 10, 1e-6, 3, 'neel', rng)
    m2 = ini_mps(6, 10, 1e-6, 3, 'neel', rng)

    assert not jnp.allclose(m1, m2)