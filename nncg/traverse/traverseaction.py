from typing import List


class TraverseAction:
    """
    Traverses the graph and doing a custom operation on visited nodes. These operations can be found in folder
    actions.
    """
    def __init__(self):
        """
        Init the class.
        """
        self.traverse_edges = None

    def pre_action(self, edge) -> bool:
        """
        This should not overwritten or used outside of Edge.traverse().
        :param edge: The edge that is currently visited.
        :return: Continue to traverse?
        """
        return self._pre_action(edge)

    def post_action(self, edge) -> bool:
        """
        This should not overwritten or used outside of Edge.traverse().
        :param edge: The edge that is currently visited.
        :return: Continue to traverse?
        """
        return self._post_action(edge)

    def _pre_action(self, edge) -> bool:
        """
        An action that can be done on the currently visited edge before the edges of edge.target are visited.
        This can be overwritten to perform a custom action defined by a user.
        :param edge: The edge that is currently visited.
        :return: Continue to traverse?
        """
        return True

    def _post_action(self, edge) -> bool:
        """
        An action that can be done on the currently visited edge after the edges of edge.target are visited.
        This can be overwritten to perform a custom action defined by a user.
        :param edge: The edge that is currently visited.
        :return: Continue to traverse?
        """
        return True


class UniqueTraverseAction(TraverseAction):
    """
    Comparable to TraverseAction but an edge is only visited once. An edge may be visited twice if the same edge
    Object is used twice in the graph. E.g. Variables are only instantiated once but used multiple times. Here
    these are only visited once.
    """
    visited: List

    def __init__(self):
        """
        Init the class.
        """
        super().__init__()
        self.visited = []

    def pre_action(self, edge) -> bool:
        """
        pre_action() of base class overwritten to ensure that an instance of an Edge Object is only
        visited once.
        :param edge: The currently visited edge
        :return: Continue to traverse?
        """
        if id(edge) in self.visited:
            return False
        self.visited.append(id(edge))
        return self._pre_action(edge)
