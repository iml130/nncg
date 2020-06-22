import pydot
from nncg.traverse.traverseaction import UniqueTraverseAction


class AddToPydot(UniqueTraverseAction):
    """
    Action to add Nodes and Edges to a pydot plot.
    """
    def __init__(self, graph):
        """
        Init this class.
        :param graph: The pydot graph to add Nodes and Edges.
        """
        super().__init__()
        self.graph = graph

    def _pre_action(self, edge) -> bool:
        """
        This action adds the Nodes and Edges to the pydot graph.
        :param edge: The currently visited Edge.
        :return: Always True.
        """

        self_name = edge.target.unique_name()
        self_node = pydot.Node(self_name, label=str(edge.target))
        owner_name = edge.owner.unique_name()
        if not '"' + owner_name + '"' in list(self.graph.obj_dict['nodes'].keys()):
            self.graph.add_node(pydot.Node(owner_name, label=str(edge.owner)))
        pydot_edge = pydot.Edge(owner_name, self_name, label=edge.get_descr())
        inverse_pydot_edge = pydot.Edge(edge.inverse.owner.unique_name(),
                                        edge.inverse.target.unique_name(),
                                        label=edge.inverse.get_descr())

        self.graph.add_node(self_node)
        self.graph.add_edge(pydot_edge)
        self.graph.add_edge(inverse_pydot_edge)

        return True
