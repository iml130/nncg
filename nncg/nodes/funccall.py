from nncg.nodes.misc import Node
from nncg.nodes.expressions import Expression


class FuncCallNode(Node):
    """
    A custom C code line, usually a function call with no assignment.
    """
    snippet = '{expr}\n'

    def __init__(self, expr: Expression, prev_node):
        """
        Init method for class.
        :param expr: The custom expression.
        :param prev_node: The previous node.
        """
        super().__init__(prev_node)
        self.expr = expr
