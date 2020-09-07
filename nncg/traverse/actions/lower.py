from nncg.traverse.traverseaction import TraverseAction


class LowerAction(TraverseAction):
    """
    Action to lower where possible
    """

    def __init__(self):
        """
        Init this class.
        """
        super().__init__()
        self.traverse_edges = lambda n: n.name_equal('next') or n.name_equal('content') or n.n_type == 'alternative'

    def _post_action(self, edge) -> bool:
        """
        If a lowering() is found in the node it is called.
        Must be a _post_action() and not _pre_action() as the graph changes with calling QuantizedNode
        and to be able to continue we must have visited the nodes.
        :param edge: The currently visited Edge.
        :return: Always True.
        """
        t = edge.get_target()
        lowering = getattr(t, "lowering", None)
        if callable(lowering):
            lowering()
        return True
