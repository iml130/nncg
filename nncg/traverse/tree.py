from __future__ import annotations
from typing import Dict, List, Optional
from copy import deepcopy
import pydot
from nncg.traverse.traverseaction import TraverseAction
from nncg.traverse.actions.addtopydot import AddToPydot


class TreeNode:
    """
    This is the base for everything in the graph. It's mainly for managing the edges dictionary.
    """
    edges: Dict[str, Edge]

    def __init__(self):
        """
        Init the class.
        """
        self.edges = dict()

    def __str__(self):
        """
        Return an (non-unique) identifier of the node for debugging etc.
        :return: A string.
        """
        return self.short_type()

    def clear_edges(self):
        """
        Remove all edges.
        :return: None.
        """
        for e in self.not_inverse_edges():
            e.remove()

    def copy(self):
        """
        Copy the node. In contrast to the Python internal copy() we want a copy of the complete node including a
        copy of the edges dictionary (two nodes must not share one). We also want copies of the edges as these
        also must not be shared but we want to stop the copying there, namely we don't want copies
        of other nodes.
        :return: The copy.
        """
        _edges = self.edges
        self.edges = {}
        n = deepcopy(self)
        self.edges = _edges
        for e in self.not_inverse_edges():
            n.add_edge(e.name, e.target, e.n_type)
        return n

    def short_type(self):
        """
        Giving a short type. It's a string of the type of the node but as Python also gives the complete 'path'
        when nodes inherit we remove this.
        :return:
        """
        n = str(type(self))
        return n[n.rindex('.') + 1:-2]

    def unique_name(self) -> str:
        """
        Giving a unique name of this node.
        :return: The name as string.
        """
        return str(self) + '(' + str(id(self)) + ')'

    def not_inverse_edges(self) -> List[Edge]:
        """
        Give all edges from this node that are not of type "inverse".
        :return: List of Edges.
        """
        return [e for e in self.edges.values() if e.n_type is not 'inverse']

    def inverse_edges(self) -> List[Edge]:
        """
        Give all edges from this node that are of type "inverse".
        :return: List of Edges
        """
        return [e for e in self.edges.values() if e.n_type == 'inverse']

    def get_node(self, name) -> TreeNode:
        """
        Get the target of the edge with this name. As the name is unique (probably due to trailing stars) it is
        a single node.
        :param name: Name of the edge.
        :return: The node that is the target of the edge.
        """
        if name in self.edges.keys():
            return self.edges[name].target
        more_results = [n for n in self.edges.keys() if '!content' in n]
        if len(more_results) == 1:
            return self.edges[more_results[0]].target
        raise

    def get_node_by_type(self, n_type) -> List[TreeNode]:
        """
        Get a List of TreeNodes that are the targets of the Edges of given type. As the type is not unique this is
        a list of all matches.
        :param n_type: Type to search for.
        :return: List of the TreeNodes found as targets.
        """
        return [self.edges[k].target for k in sorted([k for k in self.edges.keys()]) if self.edges[k].n_type is n_type]

    def has_edge(self, name):
        """
        Check if the node has an Edge of given name.
        :param name: Name of Edge.
        :return: True or False.
        """
        return self.edges.get(name) is not None

    def edge_num_by_type(self, n_type):
        """
        Same as get_node_by_type() but gives the count.
        :param n_type: Type to search for.
        :return: The count of Edges (and thus TreeNodes) found.
        """
        return len(self.get_node_by_type(n_type))

    def add_edge(self, name, target: TreeNode, n_type='forward', inverse=None, replace=False) -> Edge:
        """
        Adds an Edge to this node.
        :param name: Name of the Edge to add.
        :param target: Target TreeNode for Edge.
        :param n_type: Type of the Edge to add.
        :param inverse: An inverse is automatically added. However, here an inverse can be given.
        :param replace: Should a already existing Edge with this name be replaced? If no, a star will be added to
                        get a unique name for the new Edge.
        :return: The new Edge.
        """
        if self.edges.get(name) is not None:
            if not replace:
                return self.add_edge(name + "*", target, n_type, inverse)
            else:
                self.edges.get(name).inverse.remove()
        edge = Edge(name, target, self, n_type, inverse)
        self.edges[name] = edge
        return edge

    def search_path_end(self, edge_name) -> TreeNode:
        """
        Follow all Edges with given name till no Edge with this name can be found.
        :param edge_name: Name of the edges to follow.
        :return: The last node in path.
        """
        if edge_name in self.edges.keys():
            return self.edges[edge_name].target.search_path_end(edge_name)
        else:
            return self

    def next_node(self, name):
        """
        Get the target of the Edge with given name. Does not check if this Edge exists, if not it crashes.
        :param name: Name of the Edge.
        :return: The target TreeNode of the Edge.
        """
        return self.edges[name]

    def replace_self_with_path(self, first_node: TreeNode, last_node: TreeNode):
        """
        Searches for all Edges to target this node and then replaces this node by the first node. All outgoing
        node (not "inverse" Edges) of this node are added to the last node.
        :param first_node: Node where all incoming Edges will point to.
        :param last_node: Node to add to all outgoing Edges.
        :return: None.
        """
        edges_to_self = [e.inverse for e in self.inverse_edges()]
        for e in edges_to_self:
            e.replace_target_with_path(first_node, last_node)

    def copy_out_edges_from(self, other: TreeNode, replace=False):
        """
        Copy all Edges from other node to this node.
        :param other: The node to copy all non-inverse Edges from.
        :param replace: Replace Edge in case the name exists? If not a star is added to the name till the name is
                        unique.
        :return: None.
        """
        for e in other.not_inverse_edges():
            if e.target is not self:
                self.add_edge(e.name, e.target, e.n_type, replace=replace)

    def copy_in_edges_from(self, other: TreeNode, replace=False):
        """
        Copy the Edges pointing to other and let them point this this node.
        :param other: The other node.
        :param replace: Replace Edge in case the name exists? If not a star is added to the name till the name is
                        unique.
        :return: None.
        """
        for e in other.inverse_edges():
            if e.target is not self:
                e.target.add_edge(e.inverse.name, self, e.inverse.n_type, replace=replace)

    def remove_edge(self, name):
        """
        Remove Edge with given name. Will throw an exception if Edge cannot be found.
        :param name: Name of Edge to be removed.
        :return: None.
        """
        if self.has_edge(name):
            self.edges[name].remove()

    def remove_out_edges(self):
        """
        Remove all outgoing (non-inverse) Edges.
        :return: None.
        """
        for e in self.not_inverse_edges():
            e.remove()

    def remove_in_edges(self):
        """
        Remove all Edges pointing to this node.
        :return: None.
        """
        for e in self.inverse_edges():
            e.remove()

    def takeover_out_edges_from(self, other: TreeNode, replace=False):
        """
        Move the outgoing Edges from other TreeNode to this node.
        :param other: The other node.
        :param replace: Replace Edge in case the name exists? If not a star is added to the name till the name is
                        unique.
        :return: None.
        """
        self.copy_out_edges_from(other, replace)
        other.remove_out_edges()

    def takeover_in_edges_from(self, other: TreeNode, replace=False):
        """
        Let the Edges pointing to other TreeNode to point to this node.
        :param other: The other node.
        :param replace: Replace Edge in case the name exists? If not a star is added to the name till the name is
                        unique.
        :return: None.
        """
        self.copy_in_edges_from(other, replace)
        other.remove_in_edges()

    def takeover_edges_from(self, other, replace=False):
        """
        Combination of takeover_out_edges_from() and takeover_in_edges_from().
        :param other: The other node.
        :param replace: Replace Edge in case the name exists? If not a star is added to the name till the name is
                        unique.
        :return: None.
        """
        self.takeover_out_edges_from(other, replace)
        self.takeover_in_edges_from(other, replace)

    def merge(self, other, replace):
        """
        Merge edges from other node with this node. Basically the same as takeover_edges_from() for now, may change
        in future.
        :param other: The other node.
        :param replace: Replace Edge in case the name exists? If not a star is added to the name till the name is
                        unique.
        :return: None.
        """
        self.takeover_edges_from(other, replace)

    def traverse(self, action: TraverseAction):
        """
        Start a traverse action at this node by calling traverse on all Edges.
        :param action: The action to execute while visiting die Edges.
        :return: None.
        """
        if action.traverse_edges is None:
            edges = self.not_inverse_edges()
        else:
            edges = [self.edges[n] for n in action.traverse_edges if n in self.edges.keys()]
        for e in edges:
            e.traverse(action)

    def remove(self):
        """
        Remove this node from graph.
        :return: None.
        """
        self.remove_in_edges()
        self.remove_out_edges()

    def plot_graph(self, path):
        """
        Save a pydot plot of the graph starting at this node.
        :param path: Save the plot here.
        :return: None.
        """
        graph = pydot.Dot(graph_type='digraph')
        action = AddToPydot(graph)
        self.traverse(action)
        graph.write_png(path)


class Edge:
    """
    Class representing edges in the a directed, double linked graph. Stores additional information about the edge.
    All descriptions below include the correct handling of the inverse Edges.
    """
    n_type: str
    target: TreeNode
    owner: TreeNode
    inverse: Edge
    name: str

    def __init__(self, name, target: TreeNode, owner: Optional[TreeNode], n_type, inverse=None):
        """
        Init this class.
        :param name: Name of the node. It's in contrast to n_type a unique string information.
        :param target: Target of the directed edge.
        :param owner: The source of the directed edge.
        :param n_type: Type of the node. It's a non-unique string information.
        :param inverse: The inverse edge is not an edge in proper sense. It is comparable to the inverse edge
                        in a double linked list.
        """
        self.target = target
        self.owner = owner
        self.n_type = n_type
        self.name = name
        if inverse is None and owner is not None:
            self.add_inverse_edge()
        else:
            self.inverse = inverse

    def get_target(self):
        """
        Return the target of this Edge.
        :return: The target.
        """
        return self.target

    def add_inverse_edge(self):
        """
        Create the "inverse" Edge to this Edge.
        :return: None.
        """
        self.inverse = self.target.add_edge("!" + self.name, target=self.owner, n_type='inverse', inverse=self)

    def set_target(self, new_target: TreeNode):
        """
        Set a new target for this edge (only). Does not touch anything else.
        :param new_target: The new target.
        :return: None
        """
        self.remove()
        self.owner.add_edge(self.name, new_target, self.n_type)

    def replace_target(self, new_target: TreeNode):
        """
        Calls replace_target_with_path() with a single node as path, see replace_target_with_path() for more
        details.
        :param new_target: The new target node.
        :return: None.
        """
        self.replace_target_with_path(new_target, new_target)

    def replace_target_with_path(self, first_node: TreeNode, last_node: TreeNode):
        """
        Replaces the target of this node with first_node. Then sets for all outgoing Edges of the old target
        last_node as new owner. Does not touch other Edges pointing to the old target.
        :param first_node: First node of path as replacement.
        :param last_node: Last node of path as replacement.
        :return: None.
        """
        old_target = self.target
        self.set_target(first_node)
        last_node.takeover_out_edges_from(old_target)

    def remove(self):
        """
        Removes this Edge from graph.
        :return: None.
        """
        if self.owner.edges.get(self.name) == self:
            del self.owner.edges[self.name]
        if self.inverse.owner.edges.get(self.inverse.name) == self.inverse:
            del self.inverse.owner.edges[self.inverse.name]

    def insert_node(self, target: TreeNode):
        """
        Inserts
        :param target:
        :return:
        """
        self.insert_path(target, target)

    def insert_path(self, first_node: TreeNode, last_node: TreeNode):
        """
        Inserts between this Edge and the current target of this Edge the given path, i.e. set first_node as
        the new target and add a new Edge with the same name from last_node to current target.
        :param first_node: First node of path.
        :param last_node: Last node of path.
        :return: None.
        """
        last_node.add_edge(self.name, self.target, self.n_type)
        self.replace_target_with_path(first_node, last_node)

    def traverse(self, action):
        """
        This is the core function of traversing the graph with actions.
        Recursively traverse the graph by visiting all Edges and call pre_action and post_action of action.
        :param action: The action.
        :return: None.
        """
        if action.pre_action(self):
            self.target.traverse(action)
        action.post_action(self)

    def __str__(self):
        """
        Describes this Edge by giving the description of the target node.
        :return: String.
        """
        return str(self.target)

    def get_descr(self):
        """
        Get a description of this Edge using the name and n_type.
        :return: String.
        """
        return '{} ({})'.format(self.name, self.n_type)
