from typing import List, Dict
from nncg.traverse.tree import Edge, TreeNode
from nncg.traverse.traverseaction import TraverseAction
from nncg.nodes.expressions import Variable


class DeepCopy(TraverseAction):
    """
    This action makes a deep copy. Python has internal functions for doing deep copies but these also
    make copies of Variables (that is undesired here) and does not respect inverse edges leading to circles
    and thus an infinite copy.
    """
    old_to_new: Dict[id, TreeNode]
    excluding_edge_types: List[str] = ['inverse']

    def __init__(self):
        """
        Init this class.
        """
        super().__init__()
        self.old_to_new = {}

    def _copy_node(self, n):
        """
        Internal function to make a copy of a node. Only makes a copy if it wasn't done before, otherwise
        we would have two copies of the same Node in the new graph.
        :param n: Node to copy.
        :return: The copy.
        """
        c = self.old_to_new.get(id(n))
        if c is not None:
            return c
        c = n.copy()
        self.old_to_new[id(n)] = c
        return c

    def _follow_edge(self, edge: Edge):
        """
        Internal function to answer if the edge should be followed. It should if the target of the edge is not a
        Variable or n_type of the Edge is not in excluding_edge_types.
        :param edge: True or False.
        :return:
        """
        return edge.n_type not in self.excluding_edge_types and type(edge.target) is not Variable

    def _pre_action(self, edge: Edge) -> bool:
        """
        Here internal functions are used to make copies of the currently visited edge and target and owner.
        :param edge: The currently visited edge.
        :return: Always True.
        """
        if not self._follow_edge(edge):
            return False
        edge_from = self._copy_node(edge.owner)
        edge_to = self._copy_node(edge.target)
        edge_from.add_edge(edge.name, edge_to, edge.n_type, replace=True)
        return True

    @staticmethod
    def deep_copy(n: TreeNode) -> TreeNode:
        """
        Static method for easier application of this class. Makes a deep copy of the given Node.
        :param n: The node to deep copy.
        :return: Copy of the node.
        """
        a = DeepCopy()
        n.traverse(a)
        c = a.old_to_new.get(id(n))
        if c is None:
            return n.copy()
        return c


class DeepCopyLoop(DeepCopy):
    """
    Special version of the DeepCopy action to copy just a LoopNode with its content.
    """
    def __init__(self):
        """
        Init the action.
        """
        super().__init__()
        self.in_content = False

    def _pre_action(self, edge) -> bool:
        """
        Here we don't follow the 'next' Edge of the root LoopNode. 'next' Edges will be followed in the
        content of the LoopNode.
        :param edge: The currently visited Edge.
        :return: The return value of the overwritten node.
        """
        if edge.name == 'next' and not self.in_content:
            return False
        if edge.name == 'content':
            self.in_content = True
        return super()._pre_action(edge)

    @staticmethod
    def deep_copy(n: TreeNode) -> TreeNode:
        """
        Static method for easier application of this class.
        :param n: The node to deep copy.
        :return: Copy of the node.
        """
        a = DeepCopyLoop()
        n.traverse(a)
        return a.old_to_new[id(n)]
