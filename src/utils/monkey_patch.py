import numpy as np
from pyemma._ext.variational.estimators.covar_c import covartools
from pyemma.coordinates.data.featurization.angles import SideChainTorsions

"""
Monkey patches for pyemma. Otherwise pyemma is not compatible with python 3.11.
"""


def patched_init(self, top, selstr=None, deg=False, cossin=True, periodic=True, which="all"):
    if not isinstance(which, (tuple, list)):
        which = [which]
    if not set(which).issubset(set(self.options) | {"all"}):
        raise ValueError(
            'Argument "which" should only contain one of {}, but was {}'.format(
                ["all"] + list(self.options), which
            )
        )
    if "all" in which:
        which = self.options
    from mdtraj.geometry import dihedral

    indices_dict = {k: getattr(dihedral, "indices_%s" % k)(top) for k in which}
    if selstr:
        selection = top.select(selstr)
        truncated_indices_dict = {}
        for k, inds in indices_dict.items():
            mask = np.in1d(inds[:, 1], selection, assume_unique=True)
            truncated_indices_dict[k] = inds[mask]
        indices_dict = truncated_indices_dict

    valid = {k: indices_dict[k] for k in indices_dict if indices_dict[k].size > 0}
    if not valid:
        raise ValueError("Could not determine any side chain dihedrals for your topology!")
    self._prefix_label_lengths = np.array(
        [len(indices_dict[k]) if k in which else 0 for k in self.options]
    )
    indices = np.vstack(list(valid.values()))  # Monkey patch: Convert to list here

    super(SideChainTorsions, self).__init__(
        top=top, dih_indexes=indices, deg=deg, cossin=cossin, periodic=periodic
    )


SideChainTorsions.__init__ = patched_init


def patched_variable_cols(X, tol=0.0, min_constant=0):
    if X is None:
        return None
    from pyemma._ext.variational.estimators.covar_c._covartools import (
        variable_cols_char,
        variable_cols_double,
        variable_cols_float,
        variable_cols_int,
        variable_cols_long,
    )

    # Use bool_ instead of bool
    cols = np.zeros(X.shape[1], dtype=np.bool_, order="C")  # Changed from np.bool to np.bool_

    if X.dtype == np.float64:
        completed = variable_cols_double(cols, X, tol, min_constant)
    elif X.dtype == np.float32:
        completed = variable_cols_float(cols, X, tol, min_constant)
    elif X.dtype == np.int32:
        completed = variable_cols_int(cols, X, 0, min_constant)
    elif X.dtype == np.int64:
        completed = variable_cols_long(cols, X, 0, min_constant)
    elif X.dtype == np.bool_:  # Changed from np.bool to np.bool_
        completed = variable_cols_char(cols, X, 0, min_constant)
    else:
        raise TypeError("unsupported type of X: %s" % X.dtype)

    if completed == 0:
        return np.ones_like(
            cols, dtype=np.bool_
        )  # Monkey patch: Changed from numpy.bool to np.bool_

    return cols


covartools.variable_cols = patched_variable_cols
