from __future__ import annotations
import os
import sys
from numpy import unravel_index
from nncg.writer import Writer
from nncg.nodes.expressions import *
from nncg.traverse.tree import TreeNode
from nncg.nodes.expressions import Variable


class Writable(TreeNode):
    """
    Base class for nodes that can be written. Will be used by an traverse action.
    """
    snippet = ''

    def write_c(self):
        """
        This method is called when the WriteC action writes the C code file.
        It is called before the child nodes are visited.
        :return: None.
        """
        _exp = self.snippet.format(**self.edges, **self.__dict__)
        Writer.write_c(_exp)

    def write_c_leave(self):
        """
        LoopNodes and other nodes with 'content' nodes possibly raise the indentation and close brackets
        so these need a call when the content is left.
        :return: None.
        """
        pass


class Node(Writable):
    """
    Base class for nodes.
    """
    arch = 'general'
    var_decls = List[Variable]
    const_decls = List[Variable]
    pointer_decls = List[Variable]

    def __init__(self, prev_node: TreeNode = None, name: str = 'next'):
        """
        Init the node.
        :param prev_node: A previous node. Useful if you want a chain of nodes.
        :param name: Connects to the previous node with an edge with this name.
        """
        super().__init__()
        self.var_decls = []
        self.const_decls = []
        self.pointer_decls = []
        self.math_required = False
        if prev_node is not None:
            prev_node.add_edge(name, self)

    def get_descr(self):
        """
        Return a description of the node. Will be used to draw graphs and for debugger.
        :return:
        """
        return '{} ({})'.format(type(self), id(self))

    def match(self, node_type):
        """
        Is this node of this type?
        :param node_type: Node type to check (type()).
        :return: True or False.
        """
        return node_type is None or type(self) is node_type


class AlternativesNode(Node):
    """
    Base class for nodes that have multiple alternative paths. There will be one active path called content and this
    path is written to C code. All other paths can be selected to be active if desired. Furthermore, when performing
    optimizations and other graph traversing actions all alternatives can be processed. Afterwards can be decided, which
    one will be selected to be written.
    """
    def __init__(self, orig_node: Node, prev_node: TreeNode = None, name: str = 'next'):
        '''
        Init the node.
        :param orig_node: This node will be the initial content.
        :param prev_node: The previous node.
        :param name: Connects to the previous node with an edge with this name.
        '''
        super().__init__(prev_node, name)
        self.takeover_in_edges_from(orig_node)
        self.add_edge('content', orig_node, n_type='original')
        self.orig_node = orig_node

    def add_alternative(self, node: Node):
        '''
        Add an alternative node as content. Will not be selected as active automatically.
        :param node: The node to be added.
        :return: None.
        '''
        self.add_edge('content', node, n_type='alternative')

    def add_copy_from_orig(self):
        '''
        Copy the currently active content and add it as an alternative. Useful to perform operations on a path
        and keeping the original.
        :return: None.
        '''
        orig_copy = self.get_orig_node().copy()
        self.add_alternative(orig_copy)

    def select(self, node):
        '''
        Selects a node as active content. The node must already be an existing alternative node within this
        AlternativeNode.
        :param node: The node to be selected as active.
        :return: None
        '''
        prev_selected = self.get_node('content')
        e = self.get_edges_to(node)
        assert len(e) == 1
        e[0].remove()
        self.add_edge('content', node, replace=True)
        self.add_edge('content', prev_selected, n_type='alternative')

    def get_orig_node(self):
        '''
        Get the Node that is currently active.
        :return: The active Node.
        '''
        return self.get_node_by_type('original')[0]


class KerasLayerNode(Node):
    """
    A node that automatically tests the input of this node (output of previous node). keras_compile() will
    use the provided images (imdb) to get the results of all Keras layers. Additionally, the compiled executable
    will also do the inference on the same images and this KerasLayerNode writes the results to files. Afterwards,
    the results are compared.
    """
    in_var: Variable
    snippet = '''#ifdef CNN_TEST
{{
    FILE *f = fopen("{var_name}", "wb");
    for (int i = 0; i < {num}; i++)
        fprintf(f, "%8.8e\\n", ((float*){var_name})[i]);
    fclose(f);
}} 
#endif
'''

    def __init__(self, prev_node: Node, func, layer_name):
        """
        Init this node.
        :param prev_node: The previous node whose output will be checked.
        :param func: The Keras function executing the layer, see add_test_node().
        :param layer_name: The layer name as given by Keras.
        """
        super().__init__(prev_node)
        self.in_var = prev_node.out_var
        self.out_var = self.in_var
        self.in_dim = prev_node.out_dim
        self.out_dim = self.in_dim
        self.var_name = str(self.in_var)
        self.var_type = self.in_var.type
        self.func = func
        self.layer_name = layer_name

    def write_c(self):
        """
        This method is called when the WriteC action writes the C code file. It is called before the child nodes
        are visited. Overwritten here to set a variable.
        :return: None.
        """
        self.num = np.prod(self.in_var.dim + np.sum(self.in_var.pads, 1))
        super().write_c()

    def test(self, im, exit_on_err):
        """
        Perform the test using the provided image.
        :param im: The image as 4 dimensional array comparable to Keras.
        :return: None.
        """
        c_res = []
        with open(self.var_name) as f:
            for l in f.readlines():
                c_res.append(float(l))
        if self.func is None:
            # E.g. to just check if the input image was loaded correctly.
            res = im.reshape(*im.shape[1:])
        else:
            # Otherwise execute the Keras function.
            res = np.array(self.func([im, 0])).reshape(self.in_dim)
        c_res = np.array(c_res).reshape(self.in_var.dim + np.sum(self.in_var.pads, 1))

        if len(np.atleast_1d(self.in_var.dim)) == 3:
            # To check convolution etc.
            if self.in_var.pads[0][1] > 0:
                end_01 = -self.in_var.pads[0][1]
            else:
                end_01 = self.in_var.dim[0]
            if self.in_var.pads[1][1] > 0:
                end_11 = -self.in_var.pads[1][1]
            else:
                end_11 = self.in_var.dim[1]
            c_res = c_res[self.in_var.pads[0][0]:end_01, self.in_var.pads[1][0]:end_11, :]
        elif len(np.atleast_1d(self.in_var.dim)) == 1:
            # To check after flatten, e.g. dense layer.
            if self.in_var.pads[0][1] > 0:
                end_01 = -self.in_var.pads[0][1]
            else:
                end_01 = self.in_var.dim
            c_res = c_res[self.in_var.pads[0][0]:end_01]
        else:
            raise Exception("Unimplemented")

        if exit_on_err and not np.allclose(res, c_res,
                                           atol=0.00001 * np.max(res)):  # We have to allow a small error due to rounding errors
            print("Check of variable {} for layer {}.".format(self.in_var, self.layer_name))
            idx = unravel_index(np.argmax(res - c_res), res.shape)
            print('Largest error {} at {} ({}).'.format(np.max(res - c_res), idx, np.argmax(res - c_res)))
            print('Values: {} and {}'.format(res[idx], c_res[idx]))
            sys.exit(4)
        else:
            os.remove(self.var_name)

        return res, c_res


class ExpressionNode(Node):
    """
    A node encapsulating expressions to a node.
    """
    snippet = '{exp}\n'

    def __init__(self, exp, prev_node=None):
        """
        Init the node.
        :param exp: An expression.
        :param prev_node: The previous node.
        """
        super().__init__(prev_node)
        self.exp = exp
