from nncg.traverse.traverseaction import TraverseAction
from nncg.quantization import QuantizedNode
from nncg.nodes.misc import KerasLayerNode

import numpy as np


class QuantizeAction(TraverseAction):
    """
    Action to apply quantization where possible
    """

    def __init__(self, imdb, dtype_required):
        """
        Init this class.
        """
        super().__init__()
        self.traverse_edges = ['next']
        self.imdb = imdb
        self.max_error = 0
        self.dtype_required = dtype_required

    def _pre_action(self, edge) -> bool:
        """
        Called on every visited edge. Used to determine what values actual appear. Max and min of these values are
        used later to determine the scaling factor.
        :param edge: The edge that is currently visited.
        :return: Always True
        """
        t = edge.get_target()
        if type(t) is KerasLayerNode and callable(t.func):
            l = t.func([np.array(self.imdb), 0])
            t.out_max = np.max(l)
            t.out_min = np.min(l)
        return True

    def _post_action(self, edge) -> bool:
        """
        If a quantize() is found in the node, a QuantizedNode() is created and quatize() is called.
        Must be a _post_action() and not _pre_action() as the graph changes with calling QuantizedNode
        and to be able to continue we must have visited the nodes.
        :param edge: The currently visited Edge.
        :return: Always True.
        """
        t = edge.get_target()
        func = getattr(t, "quantize", None)
        if callable(func) and QuantizedNode.quantizable(t):
            prev_keras_node = t.get_node("!next")
            max_in = prev_keras_node.out_max
            min_in = prev_keras_node.out_min
            if min_in < 0:
                dtype = 'int8'
            else:
                dtype = 'uint8'
            if self.dtype_required == dtype:
                x_scale = QuantizedNode.quantize_scale(min_in, max_in, dtype)
                QuantizedNode(t, x_scale, prev_keras_node, dtype)
        return True
