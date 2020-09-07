from nncg.traverse.traverseaction import TraverseAction
from nncg.nodes.misc import Node
from nncg.nodes.language import CHeaderNode


class CollectVars(TraverseAction):
    """
    Action for collecting variables etc. that must be declared at beginning. Only used once in main function.
    """

    hn: CHeaderNode

    def __init__(self, hn: CHeaderNode):
        """
        Init the action.
        :param path:
        """
        super().__init__()
        self.hn = hn
        self.traverse_edges = ['content', 'next']

    def _pre_action(self, edge) -> bool:
        """
        Extend the lists by the lists in the node. See overwritten method for details.
        :param edge: The currently visited Edge.
        :return: Always True.
        """
        t = edge.target
        if issubclass(type(t), Node):
            if t.math_required:
                self.hn.math_required = True
            self.hn.pointer_decls.extend(t.pointer_decls)
            self.hn.var_decls.extend(t.var_decls)
            self.hn.const_decls.extend(t.const_decls)
        return True
