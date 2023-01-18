from .run import Run
from typing import Sequence
from pathlib import Path
import os
from .main import MAIN_PATH
from . import PACKAGE_PATH
from . import config


venv_path = Path.joinpath(PACKAGE_PATH, Path('venv/bin/activate'))
if not os.path.exists(venv_path):
    raise FileNotFoundError(f'Virtual environment {venv_path} not fount.')


def create_array_script(
        runs: Sequence[Run],
        name: str,
        hours: int,
        minutes: int,
        mem_per_cpu_mb: int,
    ):
    '''This function must be run on the machine/cluster that the script is going to be executed on.
    Otherwise paths will most likely not match.'''
    if not all(runs[0].series_name == r.series_name for r in runs):
        raise ValueError('Runs are part of different series. Only job arrays within one series can be created.')
    for r in runs:
        r.pre_execute_check()
    series_number = int(runs[0].series_name[:3])

    index_list = '{' + ','.join(r.id for r in runs) + '}'

    script = '\n'.join(
        '#!/bin/bash\n',
        f'# index list: {index_list}'
        f'#SBATCH --job-name={name}',
        '#SBATCH --qos=standard',
        '#SBATCH --mail-user=fwilde@zedat.fu-berlin.de',
        '#SBATCH --mail-type=none',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1',
        f'#SBATCH --mem-per-cpu={mem_per_cpu_mb}',
        f'#SBATCH --time={hours:02}:{minutes:02}:00\n',
        f"module add {config['PYTHON_MODULE']}",
        f'source {venv_path}',
        f'python {MAIN_PATH} {series_number}' + '${SLURM_ARRAY_TASK_ID}'
    )

    script_name = name + '-' + ''.join(r.id for r in runs) + '.sh'
    script_path = Path.joinpath(runs[0].scripts_directory(), Path(script_name))
    with open(script_path, 'x') as f:
        f.write(script)

    return index_list


def create_script(
        run: Run,
        name: str,
        hours: int,
        minutes: int,
        mem_per_cpu_mb: int,
    ):
    '''This function must be run on the machine/cluster that the script is going to be executed on.
    Otherwise paths will most likely not match.'''
    run.pre_execute_check()
    series_number = int(run.series_name[:3])

    script = '\n'.join(
        '#!/bin/bash\n',
        f'#SBATCH --job-name={name}',
        '#SBATCH --qos=standard',
        '#SBATCH --mail-user=fwilde@zedat.fu-berlin.de',
        '#SBATCH --mail-type=none',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1',
        f'#SBATCH --mem-per-cpu={mem_per_cpu_mb}',
        f'#SBATCH --time={hours:02}:{minutes:02}:00\n',
        f"module add {config['PYTHON_MODULE']}",
        f'source {venv_path}',
        f'python {MAIN_PATH} {series_number} {run.id}'
    )

    script_path = Path.joinpath(run.scripts_directory(), Path(f'{name}-{run.id}.sh'))
    with open(script_path, 'x') as f:
        f.write(script)
