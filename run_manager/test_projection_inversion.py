import pytest
from jax.typing import ArrayLike
import jax.numpy as jnp
from .projection_inversion import balls_into_bins_assignments

args = [
    (
        jnp.array([1, 1, 1, 0, 0, 0]),
        3,
        jnp.array([[1, 1, 1, 0, 0, 0]])
    ),
    (
        jnp.array([2, 1, 0]),
        4,
        jnp.zeros((0, 3))
    ),
    (
        jnp.array([2, 1, 1]),
        3,
        jnp.array([[1, 1, 1],
                   [2, 1, 0],
                   [2, 0, 1]])
    ),
    (
        jnp.array([2, 2, 2]),
        3,
        jnp.array([[2, 1, 0],
                   [2, 0, 1],
                   [1, 2, 0],
                   [0, 2, 1],
                   [1, 0, 2],
                   [0, 1, 2],
                   [1, 1, 1]])
    ),
    (
        jnp.array([1, 2, 2]),
        0,
        jnp.array([[0, 0, 0]])
    )
]


@pytest.mark.parametrize('capacities, num_balls, leaves_expected', args)
def test_balls_into_bins_assignments(
        capacities: ArrayLike,
        num_balls: int,
        leaves_expected: ArrayLike):
    leaves, _ = balls_into_bins_assignments(capacities, num_balls)

    assert all(d1 == d2 for d1, d2 in zip(leaves.shape, leaves_expected.shape))

    for cne in leaves_expected:
        # compare all to all to be independent of Hashable1DIntArray
        assert any(jnp.allclose(cne, a) for a in leaves)


args_threshold = [
    (
        jnp.array([2, 2, 2]),
        3,
        jnp.array([[1, 1, 1]]),
        1,
        0
    ),
    (
        jnp.array([2, 2, 2]),
        4,
        jnp.array([[2, 1, 1],
                   [1, 2, 1],
                   [1, 1, 2]]),
        1,
        1
    ),
    (
        jnp.array([2, 2, 1, 1]),
        3,
        jnp.array([[1, 1, 1, 0],
                   [1, 1, 0, 1],
                   [1, 0, 1, 1],
                   [0, 1, 1, 1],
                   [2, 1, 0, 0],
                   [2, 0, 1, 0],
                   [2, 0, 0, 1],
                   [1, 2, 0, 0],
                   [0, 2, 1, 0],
                   [0, 2, 0, 1],
                   ]),
        1,
        1
    )
]


@pytest.mark.parametrize(
        'capacities, num_balls, leaves_expected, threshold, max_num_bins_above_threshold',
        args_threshold
)
def test_balls_into_bins_assignments_threshold(
        capacities: ArrayLike,
        num_balls: int,
        threshold: int,
        max_num_bins_above_threshold: int,
        leaves_expected: ArrayLike):
    leaves, _ = balls_into_bins_assignments(capacities, num_balls, threshold, max_num_bins_above_threshold)

    assert all(d1 == d2 for d1, d2 in zip(leaves.shape, leaves_expected.shape))

    for cne in leaves_expected:
        # compare all to all to be independent of Hashable1DIntArray
        assert any(jnp.allclose(cne, a) for a in leaves)
