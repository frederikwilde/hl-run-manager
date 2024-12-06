import warnings
from dataclasses import dataclass
from functools import lru_cache
import re
import jax.numpy as jnp

from differentiable_tebd.physical_models.bose_hubbard_nnn import mps_evolution
from differentiable_tebd.utils.mps import mps_zero_state, add_perturbation


def _to_neel(mps, reverse=False):
    '''Convert mps_zero_state into Neel state.
    By default 101010... If reverse is False the order is 010101...'''
    start = 1 if reverse else 0
    mps = mps.at[start::2, 0, :2, 0].set(jnp.array([0., 1.]))
    return mps


@dataclass
class _Initialization:
    documentation: str
    generate: callable
    kwargs: list[str] = None

    def __call__(self, num_sites, chi, local_dim, **kwargs):
        m = mps_zero_state(num_sites, chi, None, d=local_dim)

        for k in kwargs.keys():
            if k not in self.kwargs:
                warnings.warn(f'{k} will be ignored.')

        return self.generate(m, **kwargs)


_initializations = {}


@lru_cache(maxsize=100)
def _ini_mps(num_sites, chi, local_dim, occupation):
    '''Parse the occupation identifier and return a cached (and unperturbed) initial MPS.'''
    kwargs = {}

    matches = re.findall('-reverse', occupation)
    if matches:
        occupation = ''.join(occupation.split('-reverse'))
        kwargs['reverse'] = True

    matches = re.findall(r'-start\d*', occupation)
    if matches:
        occupation = ''.join(occupation.split(matches[0]))
        kwargs['start'] = int(matches[0][6:])

    matches = re.findall(r'-time\d*.\d*', occupation)
    if matches:
        occupation = ''.join(occupation.split(matches[0]))
        kwargs['time'] = float(matches[0][5:])

    try:
        generate_mps = _initializations[occupation]
        return generate_mps(num_sites, chi, local_dim, **kwargs)
    except KeyError as e:
        raise ValueError(f'Occupation not known: {e}')


def ini_mps(num_sites, chi, mps_perturbation, local_dim, occupation, rng=None):
    # Doc string gets added after filling `_initializations`
    m = _ini_mps(num_sites, chi, local_dim, occupation)

    if mps_perturbation:
        return add_perturbation(m, mps_perturbation, rng)
    return m


### Fill _initializations dict with configurations
# half-filled


def half_filled(m, reverse=False):
    if reverse:
        s = slice(m.shape[0] // 2, None)
    else:
        s = slice(None, m.shape[0] // 2)

    m = m.at[s, 0, 0, 0].set(0.)
    m = m.at[s, 0, 1, 0].set(1.)
    return m


_initializations['half-filled'] = _Initialization(
    'Only the left half of the system is filled, each site with one particle.',
    half_filled,
    ['reverse']
)

# neel

_initializations['neel'] = _Initialization(
    'Every other site is filled, beginning with a filled site. 101010...',
    _to_neel,
    ['reverse']
)

# neel-one-third


def neel_one_third(m, start=0):
    m = m.at[start::3, 0, 0, 0].set(0)
    m = m.at[start::3, 0, 1, 0].set(1)
    return m


_initializations['neel-one-third'] = _Initialization(
    '100100..., where `start` marks the position of the first 1',
    neel_one_third,
    ['start']
)

# dimer


def dimer(m):
    s = .5 ** (1/4)

    for i in range(0, m.shape[0], 2):
        m = m.at[i, 0, 0, 0].set(1j * s)
        m = m.at[i, 0, 1, 1].set(-s)
        m = m.at[i+1, 0, 0, 0].set(0)
        m = m.at[i+1, 0, 1, 0].set(-s)
        m = m.at[i+1, 1, 0, 0].set(-s)

    return m


_initializations['dimer'] = _Initialization(
    (
        'A Neel state which has evolved up to time pi/4 under '
        'nearest-neighbor hopping between disjunct pairs of sites '
        '(0-1, 2-3, 4-5, ... but not 1-2, 3-4, ...) '
        'One dimer is given by: 1/sqrt(2) * (|10> - i|01>)'
    ),
    dimer
)

# quench


def quench(m, reverse=False, time=jnp.pi/8):
    '''Quench with J1=0.2, J2=0.01, U=1. for time T.'''
    m = _to_neel(m, reverse)
    params = jnp.array([0.2, 0.01, 1.] + len(m) * [0.])
    m, _ = mps_evolution(params, time/100, 100, m)
    return m


_initializations['quench'] = _Initialization(
    (
        'A quench of the Neel state with the following '
        'parameters: J1 = 0.2, J2 = 0.01, U = 1. The Neel state is 101010... '
        'and if `rev` is specified, it is reversed, i.e., 010101...'
    ),
    quench,
    ['reverse', 'time']
)


# Doc string of `ini_mps`

ini_mps.__doc__ = '''Creates specific MPS, such as the Neel state and others.

Some states, such as the Neel state, can be reversed in order. I.e., instead
of 010101, by adding `'-rev'` into the `occupation` description, we get
101010 for `'neel-rev'`.

Args:
    num_sites (int)
    chi (int)
    mps_perturbation (float)
    local_dim (int)
    occupation (str): Available particle occupation configurations:
''' + \
'\n'.join(f'\t{k}: {v.documentation}' for k, v in _initializations.items()) + \
'\n\n\tAdditionally, the arguments `-reverse`, `-start`, and `-time` can be added to some of the above.'
