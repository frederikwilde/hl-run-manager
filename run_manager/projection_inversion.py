from jax.typing import ArrayLike
import jax
import jax.numpy as jnp


@jax.jit
def int_array_hash(array):
    return (10 ** jnp.arange(len(array), dtype=jnp.int32)).dot(array)


class Hashable1DIntArray:
    def __init__(self, array):
        if len(array.shape) != 1:
            raise ValueError('Only 1D arrays implemented.')
        if not jnp.issubdtype(array.dtype, jnp.integer):
            raise ValueError('Only integer arrays should be hashed.')

        self.array = array

    def __repr__(self):
        return f'Hashable1DIntArray({self.array})'

    def __hash__(self):
        return int(int_array_hash(self.array))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f'Equality can only be checked against {type(self)}. Got type {type(other)}.')

        return jnp.all(self.array == other.array)


def balls_into_bins_assignments(
    capacities: ArrayLike,
    num_balls: int,
    threshold: int = None,
    max_num_bins_above_threshold: int = None
):
    '''Breadth-first search for all possible assignments of a specified number of balls
    into bins represented by elements of `capacities`. Bin `i` can hold at most
    `capacities[i]` balls.'''
    zeros = Hashable1DIntArray(jnp.zeros(len(capacities), dtype=jnp.int32))
    queue = [zeros]
    seen_nodes = set([zeros])
    leaves = set()
    leaves_encountered = 0

    if num_balls == 0:
        return jnp.zeros((1, len(capacities)), dtype=jnp.int32), 1

    while queue:
        assignment = queue.pop(0)
        next_node_is_leave = (jnp.sum(assignment.array) + 1 == num_balls)

        for i, diff in enumerate(capacities - assignment.array):
            if diff <= 0:
                # bin i is at full capacity
                continue

            new_assignment = assignment.array.at[i].add(1)

            if threshold is not None:
                num_bins_above_threshold = jnp.sum(new_assignment > threshold)
                if num_bins_above_threshold > max_num_bins_above_threshold:
                    # skip since it has too many highly occupied bins
                    continue

            new_assignment = Hashable1DIntArray(new_assignment)

            if next_node_is_leave:
                leaves.add(new_assignment)
                leaves_encountered += 1
                continue

            if new_assignment not in seen_nodes:
                queue.append(new_assignment)
                seen_nodes.add(new_assignment)

    return jnp.array([n.array for n in leaves]), leaves_encountered


def measurements_compatible_with_projection(
    parity_projected_measurement,
    fock_space_truncation,
    num_particles_total
):
    capacities = (fock_space_truncation - 1 - parity_projected_measurement) // 2
    # the number of pairs of particles that still fit into each site

    num_particles_free = (num_particles_total - jnp.sum(parity_projected_measurement))
    if num_particles_free % 2 != 0:
        raise ValueError(
            f'Number of particles mismatch in {parity_projected_measurement} '
            f'with {num_particles_total} particles expected in total.'
        )

    num_pairs_free = num_particles_free // 2
    pair_assignments, counter = balls_into_bins_assignments(
        capacities,
        num_pairs_free,
    )

    return parity_projected_measurement + 2 * pair_assignments, counter
