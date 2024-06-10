from collections import namedtuple
from typing import Sequence
from datetime import datetime, timedelta
import re
from tabulate import tabulate
from sqlalchemy import ScalarResult

from run_manager.run import Status


Column = namedtuple('Column', ['attribute', 'name', 'filter'], defaults=[None, None, None])


def table(runs: ScalarResult, columns: Sequence[Column], tablefmt='html'):
    """Easily modifiable HTML table to quickly print out query results"""

    body = []
    for r in runs:
        row = []
        for col in columns:
            content = getattr(r, col.attribute)
            row.append(col.filter(content) if col.filter else content)

        body.append(row)

    print(f"{len(body)} runs")
    return tabulate(body, headers=[col.name for col in columns], tablefmt=tablefmt)


def _str_to_seconds(time_string: str) -> float:
    if '-' in time_string:
        t = datetime.strptime(time_string, '%d-%H:%M:%S')
        dt = timedelta(days=t.day, hours=t.hour, minutes=t.minute, seconds=t.second)
    else:
        t = datetime.strptime(time_string, '%H:%M:%S')
        dt = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    return dt.total_seconds()


def duration_memory(run):
    """Extract time (sec.) and memory (MB) from Slurm data."""
    if run.status in [Status.NOT_IN_DB, Status.IN_DB, Status.JOB_STARTED]:
        return None

    stats = run.slurm_stats()
    stats_rows = stats.split('\n')

    duration = max([
        _str_to_seconds(t)
        for t in re.findall(r'(?:\d{1,2}(?=-)|\b)-{0,1}\d{2}:\d{2}:\d{2}', stats)
    ])

    match = next(re.finditer('AveVMSize', stats_rows[0]))
    ave_vm_size_position = match.end()

    memory_values = []
    for row in stats_rows:
        matches = re.findall(r'[0-9]*.[0-9]*M$', row[:ave_vm_size_position])
        if matches:
            memory_values.append(float(matches[0][:-1]))

    if memory_values:
        return duration, max(memory_values)
