from nncg.nodes.misc import Node
from nncg.nodes.controlflow import LoopNode
from nncg.allocation import Allocation
from nncg.nodes.expressions import IndexedVariable, Constant
from nncg.nodes.arithmetic import MultNode
from nncg.nodes.misc import AlternativesNode


class QuantizedNode(AlternativesNode):
    def __init__(self, node: Node, x_scale, prev_keras_node, dtype):
        """
        Init the node.
        :param node: The node to be quantized.
        :param x_scale: Scaling factor from a previous quantize_scale() to scale weights and bias.
        :param prev_keras_node: The last KerasLayerNode. Required for quantization.
        :param dtype: Target data type to quantize to.
        """
        super().__init__(node)
        self.takeover_out_edges_from(node)
        node = node.copy()
        q_node = QuantizeNode(x_scale, prev_keras_node, dtype)
        node.in_var = q_node.out_var
        self.add_alternative(q_node)
        q_node.add_edge('next', node)
        node.quantize(x_scale)
        w_scale = node.scale
        next_node = self.get_node('next')
        deq_node = DequantizeNode(w_scale, x_scale, next_node, node)
        node.add_edge('next', deq_node)
        assert next_node.in_dim == deq_node.out_dim
        self.get_node('next').in_var = deq_node.out_var


    @staticmethod
    def quantizable(node: Node):
        """
        Is this Node quantizable? Should ask before instantiating this.
        :param node: The Node to ask for.
        :return:
        """
        prev_keras_node = node.get_node("!next")
        return getattr(prev_keras_node, "out_max", None) is not None

    @staticmethod
    def quantize_scale(min, max, type):
        """
        Given the range for possible values and the desired type for quantization, get the scaling factor.
        :param min: Minimum possible value.
        :param max: Maximum possible value.
        :param type: Desired type as target for quantization.
        :return: The scaling factor.
        """
        if abs(min) > abs(max):
            v = abs(min)
        else:
            v = abs(max)
        if type == 'int8':
            return v / 127
        elif type == 'uint8':
            return v / 256


class QuantizeNode(Node):
    """
    Class to quantize a Node. It is placed before the
    """
    def __init__(self, x_scale, prev_node, dtype):
        """
        Init this Node.
        :param x_scale: The scale previously determined with quantize_scale().
        :param prev_node: The previous node.
        :param dtype: The target data type for quantization.
        """
        super().__init__()
        self.in_var = prev_node.out_var
        self.in_dim = prev_node.out_dim
        self.out_dim = self.in_dim
        self.out_var = Allocation.allocate_var(dtype, 'x', self.out_dim)
        self.out_var.change_padding(self.in_var.pads)
        self.x_scale = x_scale

    def lowering(self):
        """
        Create the Nodes required to express this node in ANSI C code. It actually creates loops to convert
        all floats to the desired data type applying the given scale.
        This loop will stay in graph to provide meta information.
        :return: None.
        """
        loops, idxs = LoopNode.create_loops(self.in_var.dim)
        in_var_idx = IndexedVariable(self.in_var)
        out_var_idx = IndexedVariable(self.out_var)
        in_var_idx.set_indices(idxs)
        out_var_idx.set_indices(idxs)
        div_node = MultNode(out_var_idx, in_var_idx, Constant(1 / self.x_scale))
        loops[-1].add_edge('content', div_node)
        self.add_edge('content', loops[0])
        self.var_decls.append(self.out_var)


class DequantizeNode(Node):
    """
    Class for converting quantized values to float again.
    """
    def __init__(self, const_scale, x_scale, next_node, prev_node):
        """
        Init the node.
        :param const_scale: Scaling factor used for quantizing the constant weights.
        :param x_scale: Scaling factor used for quantizing the input layer.
        :param prev_node: The previous node.
        """
        super().__init__()
        self.in_var = prev_node.out_var
        self.in_dim = prev_node.out_dim
        self.out_dim = self.in_dim
        self.out_var = next_node.in_var
        self.x_scale = x_scale
        self.const_scale = const_scale

    def lowering(self):
        """
        Create the Nodes required to express this node in ANSI C code. It actually creates loops to convert
        all quantized values back to floats.
        This loop will stay in graph to provide meta information.
        :return: None.
        """
        loops, idxs = LoopNode.create_loops(self.in_var.dim)
        in_var_idx = IndexedVariable(self.in_var)
        out_var_idx = IndexedVariable(self.out_var)
        in_var_idx.set_indices(idxs)
        out_var_idx.set_indices(idxs)
        div_node = MultNode(out_var_idx, in_var_idx, Constant(self.x_scale * self.const_scale))
        loops[-1].add_edge('content', div_node)
        self.add_edge('content', loops[0])
