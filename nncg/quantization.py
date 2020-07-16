from nncg.nodes.misc import Node
from nncg.nodes.controlflow import LoopNode
from nncg.allocation import Allocation
from nncg.nodes.expressions import IndexedVariable, Constant
from nncg.nodes.arithmetic import MultNode
from nncg.nodes.language import CHeaderNode

import numpy as np


class QuantizedNode(Node):
    def __init__(self, node: Node, x_scale, prev_keras_node):
        super().__init__()

        node.replace_self_with_path(self, self)
        q_node = QuantizeNode(x_scale, prev_keras_node)
        node.in_var = q_node.out_var
        self.add_edge('content', q_node)
        q_node.add_edge('next', node)
        w_scale = node.scale
        deq_node = DequantizeNode(w_scale, x_scale, node)
        node.add_edge('next', deq_node)
        assert self.get_node('next').in_dim == deq_node.out_dim
        self.get_node('next').in_var = deq_node.out_var


    @staticmethod
    def quantizable(node: Node):
        prev_keras_node = node.get_node("!next")
        return getattr(prev_keras_node, "out_max", None) is not None

    @staticmethod
    def quantize_scale(min, max, type):
        if abs(min) > abs(max):
            v = abs(min)
        else:
            v = abs(max)
        if type == 'int8':
            return v / 127
        elif type == 'uint8':
            return v / 256


class QuantizeNode(Node):
    def __init__(self, x_scale, prev_node):
        super().__init__()
        self.in_var = prev_node.out_var
        self.in_dim = prev_node.out_dim
        self.out_dim = self.in_dim
        self.out_var = Allocation.allocate_var('uint8', 'x', self.out_dim)
        self.out_var.change_padding(self.in_var.pads)
        self.x_scale = x_scale

    def lowering(self):
        loops, idxs = LoopNode.create_loops(self.in_var.dim)
        in_var_idx = IndexedVariable(self.in_var)
        out_var_idx = IndexedVariable(self.out_var)
        in_var_idx.set_indices(idxs)
        out_var_idx.set_indices(idxs)
        div_node = MultNode(out_var_idx, in_var_idx, Constant(1 / self.x_scale))
        loops[-1].add_edge('content', div_node)
        self.add_edge('content', loops[0])
        CHeaderNode.instance().var_decls.append(self.out_var)


class DequantizeNode(Node):
    def __init__(self, const_scale, x_scale, prev_node):
        super().__init__()
        self.in_var = prev_node.out_var
        self.in_dim = prev_node.out_dim
        self.out_dim = self.in_dim
        self.out_var = Allocation.allocate_var('float', 'x', self.out_dim)
        self.x_scale = x_scale
        self.const_scale = const_scale

    def lowering(self):
        loops, idxs = LoopNode.create_loops(self.in_var.dim)
        in_var_idx = IndexedVariable(self.in_var)
        out_var_idx = IndexedVariable(self.out_var)
        in_var_idx.set_indices(idxs)
        out_var_idx.set_indices(idxs)
        div_node = MultNode(out_var_idx, in_var_idx, Constant(self.x_scale * self.const_scale))
        loops[-1].add_edge('content', div_node)
        self.add_edge('content', loops[0])
        CHeaderNode.instance().var_decls.append(self.out_var)
