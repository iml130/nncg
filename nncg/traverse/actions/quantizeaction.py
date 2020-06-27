from nncg.traverse.traverseaction import TraverseAction
from nncg.quantization import QuantizedNode


class QuantizeAction(TraverseAction):
    """
    Action to apply quantization where possible
    """
    def __init__(self):
        """
        Init this class.
        """
        super().__init__()
        self.traverse_edges = ['next']

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
        if callable(func):
            QuantizedNode(t)
            func()
        return True