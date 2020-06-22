from typing import List
from nncg.traverse.tree import TreeNode
from nncg.traverse.traverseaction import TraverseAction


class ReplaceExpression(TraverseAction):
    """
    This action replaces an Expression (or in general a TreeNode) with another Expression (or again a TreeNode).
    It searches the graph and replaces all occurrences.
    """
    expr_to_replace: TreeNode
    replacement_expr: TreeNode
    inserted_node: List[id]

    def __init__(self, expr_to_replace: TreeNode, replacement_expr: TreeNode):
        """
        Init the action.
        :param expr_to_replace: The Expression to be replaced. Could also be a Variable, Constant etc.
        :param replacement_expr: The replacement Expression, Variable, Constant etc.
        """
        super().__init__()
        self.expr_to_replace = expr_to_replace
        self.replacement_expr = replacement_expr
        self.inserted_edges = []

    def _pre_action(self, edge) -> bool:
        """
        Called for every visited node. It then replaces if the the target is the desired node.
        :param edge: The currently visited edge.
        :return: None.
        """
        if edge.get_target() == self.replacement_expr:
            return False
        if edge.get_target() == self.expr_to_replace:
            edge.replace_target(self.replacement_expr)
        return True
