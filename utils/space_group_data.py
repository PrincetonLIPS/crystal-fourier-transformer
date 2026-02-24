"""
Cached loader for pre-extracted space group and plane group symmetry data.
The data file (data/space_group_data.npz) contains the raw symmetry operations
and basis matrices for all 230 space groups and 17 plane groups.
"""

import os
import numpy as np
from functools import lru_cache

_DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'space_group_data.npz')


@lru_cache(maxsize=1)
def _load_data():
    path = os.path.abspath(_DATA_FILE)
    return dict(np.load(path))


def get_space_group_operations(sg_number):
    """Return (operations, basis) for a space group (1-230).

    operations: list of (4, 4) numpy arrays (only the valid ones, not padded)
    basis: (3, 3) numpy array
    """
    data = _load_data()
    idx = sg_number - 1
    n_ops = int(data['sg_op_counts'][idx])
    operations = [data['sg_operations'][idx, i] for i in range(n_ops)]
    basis = data['sg_bases'][idx]
    return operations, basis


def get_plane_group_operations(pg_number):
    """Return (operations, basis) for a plane group (1-17).

    operations: list of (3, 3) numpy arrays (only the valid ones, not padded)
    basis: (2, 2) numpy array
    """
    data = _load_data()
    idx = pg_number - 1
    n_ops = int(data['pg_op_counts'][idx])
    operations = [data['pg_operations'][idx, i] for i in range(n_ops)]
    basis = data['pg_bases'][idx]
    return operations, basis


def get_all_space_group_data():
    """Return the full raw arrays for all 230 space groups.

    Returns:
        sg_operations: (230, 192, 4, 4)
        sg_op_counts: (230,)
        sg_bases: (230, 3, 3)
    """
    data = _load_data()
    return data['sg_operations'], data['sg_op_counts'], data['sg_bases']
