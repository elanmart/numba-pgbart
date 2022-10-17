from typing import Any

import numpy as np
from numba import float64 as f64
from numba import int64 as i64
from numba.core.types import DictType
from numba.experimental import jitclass
from numba.typed.typedlist import List


def NumbaType(jit_cls) -> Any:
    return jit_cls.class_type.instance_type


@jitclass(
    spec=[
        ("value", f64),
        ("index", i64),
        ("split_idx", i64),
    ]
)
class Node:
    def __init__(self):
        self.index = 0
        self.value = 0.0
        self.split_idx = 0

    def init_leaf(self, index, value):
        self.index = index
        self.value = value
        self.split_idx = -1

        return self

    def init_internal(self, index, value, split_idx):
        self.index = index
        self.value = value
        self.split_idx = split_idx

        return self

    def clone(self):
        new = Node()
        new.index = self.index
        new.value = self.value
        new.split_idx = self.split_idx

        return new

    def is_leaf(self) -> bool:
        return self.split_idx < 0

    def is_internal(self) -> bool:
        return self.split_idx >= 0

    def left(self):
        return self.index * 2 + 1

    def right(self):
        return self.index * 2 + 2

    def depth(self):
        ret = np.floor(np.log2(self.index + 1))

        return i64(ret)


@jitclass(
    spec=[
        ("nodes", DictType(i64, NumbaType(Node))),  # type: ignore
    ]
)
class Tree:
    def __init__(self):
        self.nodes = {}

    def init(self, root_value):
        root = Node().init_leaf(0, root_value)
        self.nodes[0] = root

        return self

    def clone(self):
        new = Tree()
        new.nodes = {idx: node.clone() for idx, node in self.nodes.items()}

        return new

    def root(self) -> Node:
        return self.nodes[0]

    def get_node(self, idx: int) -> Node:
        return self.nodes[idx]

    def set_node(self, node: Node) -> "Tree":
        self.nodes[node.index] = node

        return self

    def update_leaf_value(self, idx, value):
        node = self.get_node(idx)
        node.value = value

    def split_leaf(
        self,
        idx: int,
        split_idx: int,
        split_value: float,
        left_value: float,
        right_value: float,
    ):
        parent = Node().init_internal(idx, split_value, split_idx)
        left = Node().init_leaf(parent.left(), left_value)
        right = Node().init_leaf(parent.right(), right_value)

        self.set_node(parent).set_node(left).set_node(right)

        return left.index, right.index

    def predict(self, x):
        node = self.root()
        while node.is_internal():
            if x[node.split_idx] >= node.value:
                node = self.get_node(node.left())
            else:
                node = self.get_node(node.right())

        return node.value

    def get_split_variables(self) -> List:

        ret = List.empty_list(i64)

        for node in self.nodes.values():
            if node.is_internal():
                ret.append(node.split_idx)

        return ret  # type: ignore
