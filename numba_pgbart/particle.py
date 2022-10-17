from typing import Any

import numpy as np
from numba import float64 as f64
from numba import int64 as i64
from numba import uint8 as u8
from numba.core.types import DictType, ListType
from numba.experimental import jitclass
from numba.typed.typedlist import List

from .tree import Node, Tree

NOTHING = u8(0)


def NumbaType(jit_cls) -> Any:
    return jit_cls.class_type.instance_type


@jitclass(spec=[("items", ListType(i64)), ("idx", i64)])
class IntDeque:
    """Extremely simple implementation of Deque, which is not supported in numba.
    This is very wasteful in terms of memory, but I think it's ok for our use case.

    It is also only capable of storing integers.
    """

    def __init__(self):
        self.items = List.empty_list(i64)
        self.idx = 0

    def init(self):
        return self

    def clone(self):
        ret = IntDeque()
        ret.items = self.items.copy()
        ret.idx = self.idx

        return ret

    def append(self, item):
        self.items.append(item)

    def pop_left(self):
        item = self.items[self.idx]
        self.idx += 1
        return item

    def empty(self):
        return self.idx == len(self.items)


@jitclass(spec=[("n_points", i64), ("n_covars", i64), ("kfactor", f64)])
class ParticleParams:
    def __init__(self):
        self.n_points = 0
        self.n_covars = 0
        self.kfactor = 0.0

    def init(self, n_points, n_covars, kfactor):
        self.n_points = n_points
        self.n_covars = n_covars
        self.kfactor = kfactor

        return self

    def clone(self):
        return ParticleParams().init(self.n_points, self.n_covars, self.kfactor)

    def with_new_kf(self, kfactor):
        return ParticleParams().init(self.n_points, self.n_covars, kfactor)


@jitclass(
    spec=[
        ("leaf_nodes", DictType(i64, u8)),
        ("expansion_nodes", NumbaType(IntDeque)),
        ("data_indices", DictType(i64, i64[:])),
    ]
)
class Indices:
    def __init__(self):
        self.leaf_nodes = {0: NOTHING}
        self.expansion_nodes = IntDeque()
        self.data_indices = {0: np.arange(0)}

    def init(self, n):
        self.data_indices[0] = np.arange(n)

        return self

    def clone(self):
        new = Indices()
        new.leaf_nodes = self.leaf_nodes.copy()
        new.data_indices = self.data_indices.copy()
        new.expansion_nodes = self.expansion_nodes.clone()

        return new

    def empty(self):
        return self.expansion_nodes.empty()

    def add_index(self, idx, data_rows):
        self.leaf_nodes[idx] = NOTHING
        self.data_indices[idx] = data_rows
        self.expansion_nodes.append(idx)

    def pop_expansion_index(self):
        return self.expansion_nodes.pop_left()

    def remove_index(self, idx):
        self.leaf_nodes.pop(idx)
        self.data_indices.pop(idx)


@jitclass(spec=[("log_w", f64), ("log_likelihood", f64)])
class Weight:
    def __init__(self):
        self.log_w = 0.0
        self.log_likelihood = 0.0

    def init(self):
        return self

    def clone(self):
        new = Weight()
        new.log_w = self.log_w
        new.log_likelihood = self.log_likelihood

        return new

    def reset(self, log_likelihood):
        self.log_w = log_likelihood
        self.log_likelihood = log_likelihood

    def udpate(self, log_likelihood):
        log_w = self.log_w + log_likelihood - self.log_likelihood

        self.log_w = log_w
        self.log_likelihood = log_likelihood


@jitclass(
    spec=[
        ("params", NumbaType(ParticleParams)),
        ("tree", NumbaType(Tree)),
        ("indices", NumbaType(Indices)),
        ("weight", NumbaType(Weight)),
    ]
)
class Particle:
    def __init__(self):
        self.params = ParticleParams()
        self.tree = Tree()
        self.indices = Indices()
        self.weight = Weight()

    def init(self, params, predicted_value):
        self.params = params.clone()
        self.tree = self.tree.init(predicted_value)
        self.indices = self.indices.init(params.n_points)
        self.weight = self.weight.init()

    def clone(self):
        new = Particle()
        new.tree = self.tree.clone()
        new.indices = self.indices.clone()
        new.weight = self.weight.clone()

        return new

    def with_resampled_leaves(self, sampler_state):
        ret = self.clone()

        for leaf_idx in self.indices.leaf_nodes:

            if leaf_idx == 0:
                continue

            data_indices = self.indices.data_indices[leaf_idx]
            value = sampler_state.sample_leaf_value(data_indices)
            self.tree.update_leaf_value(leaf_idx, value)

        return ret

    def grow(self, X, sampler_state):
        if self.indices.empty():
            return False

        idx = self.indices.pop_expansion_index()
        leaf = self.tree.get_node(idx)
        expand = sampler_state.probabilities.sample_expand_flag(leaf.index)

        if not expand:
            return False

        rows = self.indices.data_indices[idx]
        split_idx = sampler_state.probabilites.sample_split_index()
        feature_values = X[rows, split_idx]

        if len(feature_values) == 0:
            return False

        split_value = sampler_state.probabilites.sample_split_value(feature_values)
        (l_inds, r_inds) = self.split_data(rows, feature_values, split_value)

        (l_val, r_val) = (
            self.leaf_value(l_inds, sampler_state),
            self.leaf_value(r_inds, sampler_state),
        )

        (l_idx, r_idx) = self.tree.split_leaf(idx, split_idx, split_value, l_val, r_val)
        self.indices.remove_index(idx)

        self.indices.add_index(l_idx, l_inds)
        self.indices.add_index(r_idx, r_inds)

        return True

    def predict(self):
        y_hat = np.empty((self.params.n_points, ), dtype=np.float64)
        
        for idx in self.indices.leaf_nodes:
            leaf = self.tree.get_node(idx)
            row_inds = self.indices.data_indices[idx]
            for i in row_inds:
                y_hat[i] = leaf.value

        return y_hat

    def split_data(self, row_indices, feature_values, split_value):
        left_indices = row_indices[feature_values <= split_value]
        right_indices = row_indices[feature_values > split_value]
        
        return left_indices, right_indices

    def leaf_value(self, data_indices, sampler_state):
        node_preds = sampler_state.predictions_subset(data_indices)
        mu = np.mean(node_preds)
        value = sampler_state.probabilites.sample_leaf_value(mu, self.params.kfactor)

        return value

    def finished(self):
        return self.indices.empty()

    def split_variables(self):
        return self.tree.get_split_variables()
