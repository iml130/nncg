from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
from nncg.tools import _len
from nncg.traverse.tree import TreeNode


class Expression(TreeNode):
    """
    Node to express a general expression. It is usually used as part of Arithmetic oder meta
    nodes.
    """
    snippet = ''

    def __init__(self, snippet, **kwargs):
        """
        Init the node.
        :param snippet: The C code snippet with {} to be replace by further Expressions or Variables. E.g.
                        {var} / {stride0}
        :param kwargs:  A dictionary providing the data required for the above snippet. In the example above it would
                        be {'var': <Variable>, 'stride0': 1} for example.
        """
        super().__init__()
        self.snippet = snippet
        for a in kwargs:
            self.add_edge(a, kwargs[a], 'm_expr')

    def __str__(self):
        """
        Returns this Expression as a string.
        :return: The string.
        """
        return self.snippet.format(**self.edges)


class Constant(TreeNode):
    """
    Simple node just representing a constant.
    """
    def __init__(self, c):
        """
        Init this node.
        :param c: Arbitrary data that can return a string of itself.
        """
        super().__init__()
        self.c = c

    def __str__(self):
        """
        Get this Constant as a string.
        :return: The string.
        """
        return str(self.c)

    def get_type(self):
        """
        Get the type of this constant. It's always float.
        :return: float
        """
        return 'float'



class Variable(TreeNode):
    """
    Node representing a Variable. It can be a scalar value like (in C notation) float or int but also an array.
    In case of an array this Node will return the name of the array variable without indices. In the following
    we assume as an example that we want an array "float matrix[3][3]".
    For arrays a padding can be set. These increase the size of the array on declaration but does not affect
    this Variable elsewhere. The purpose is to enable different
    """
    type: str
    name: str
    dim: List
    alignment: str
    pads: List[List[int]] = None
    init_data: None

    def __init__(self,
                 type: str,
                 name: str,
                 dim: Optional[List[int]],
                 alignment: int,
                 index, init_data=None):
        """
        Init the Variable.
        :param type: Type as string, e.g. "float", "int" etc.
        :param name: Name of Variable. Index will be added as a number to get a unique name.
        :param dim: Dimensions in case of an array. None if no array.
        :param alignment: Desired alignment in bytes. Can be changed later. 0 means no alignment required.
        :param index: Number to get a unique name.
        :param init_data: Initial data. Can later be written into the C file.
        """
        super().__init__()
        self.decl_written = False
        self.index = index
        self.type = type
        self.name = name
        self.dim = dim
        self.set_alignment(alignment)
        self.init_data = init_data
        self.pads = _len(dim) * [[0, 0]]
        self.temporal_value = None

    @staticmethod
    def type_to_c(t) -> str:
        '''
        Internal type name to C type name.
        :param t: The internal name.
        :return: The C style name.
        '''
        type_map = {
            'float': 'float',
            'float32': 'float',
            'float64': 'double',
            'int8': 'int8_t',
            'uint8': 'unsigned char',
            'int16': 'int16_t',
            '__m128i': '__m128i',
            'int': 'int'
        }
        return type_map[str(t)]

    @staticmethod
    def type_to_width(t):
        '''
        Give the bit width of the internal type name.
        :param t: The internal type name.
        :return: The bit width.
        '''
        width_map = {
            'float': 32,
            'float32': 32,
            'float64': 64,
            'int8': 8,
            'uint8': 8,
            'int16': 16,
            'int': 32
        }
        return width_map[str(t)]

    def __str__(self):
        """
        Get name of Variable (including unique number).
        :return: The string.
        """
        if self.temporal_value is not None:
            return str(self.temporal_value)
        return '{name}_{index}'.format(name=self.name, index=self.index)

    def change_padding(self, pads: List[List[int]]):
        """
        Set a different padding size.
        :param pads: New padding.
        :return: None.
        """
        assert len(pads) == _len(self.dim)
        self.pads = pads

    def get_cast(self):
        """
        Get the string to cast something to the type of this variable.
        :return: The cast string.
        """
        return '({}*)'.format(self.type)

    def get_type(self):
        """
        Return the type of the data.
        :return: The type
        """
        return self.type

    def _get_dim_str(self):
        """
        Get the string for defining an array.
        :return: The string.
        """
        if self.dim is None:
            return ''
        return ''.join(['[' + str(i + j[0] + j[1]) + ']' for i, j in zip(np.atleast_1d(self.dim), self.pads)])

    @staticmethod
    def format_value(v, dtype: np.dtype):
        '''
        Give a string for writing this value.
        :param v: The value.
        :param dtype: The datatype of the value.
        :return: The formatted string.
        '''
        if dtype == 'float32':
            return np.format_float_scientific(v, precision=15)
        elif dtype == 'int8':
            return str(v)
        elif dtype == 'int16':
            return str(v)
        else:
            raise Exception("Unknown data type.")

    def get_def(self, write_init_data=True):
        """
        Get the string to define this Variable. Primarily useful for CHeaderNode.
        :param write_init_data: Should also the data be written into the C file for initialization?
        :return: The string.
        """
        if self.decl_written:
            return
        self.dim_str = self._get_dim_str()
        if self.init_data is not None and write_init_data:
            self.data_str = ','.join([Variable.format_value(f, self.init_data.dtype)
                                      for f in (self.init_data.flatten())])
        else:
            self.data_str = '0'
        self.var_type = Variable.type_to_c(self.type)
        return 'static {var_type} {name}_{index} {alignment} {dim_str} = {{ {data_str} }};\n'.format(**self.__dict__)

    def get_pointer_decl(self):
        """
        This returns a string to declare this Variable as a pointer.
        :return: The declaration.
        """
        return '{type} *{name}_{index} {alignment};\n'.format(**self.__dict__)

    def set_alignment(self, bytes):
        """
        Set a new alignment.
        :param bytes: Address must be dividable by this number. 0 for no alignment.
        :return: None.
        """
        if bytes > 0:
            self.alignment = 'alignas({})'.format(8 * bytes)
        else:
            self.alignment = ''


class IndexedVariable(TreeNode):
    """
    This extension to a variable adds array indices to it ("[]").
    """
    def __init__(self, var, padding_to_offset=True):
        '''
        Init this IndexVariable.
        :param var: The Variable to add indices.
        :param padding_to_offset: If this is True, the padding will be bypassed by adding an offset to
                                  all accesses. Useful if a Variable later needs padding but this layer not so
                                  the padding is already added but bypassed here.
        '''
        super().__init__()
        self.add_edge('var', var)
        self.padding_to_offset = padding_to_offset

    def get_type(self):
        """
        Return the type of the Variable that is indexed here.
        :return: The type as string.
        """
        return self.get_node('var').get_type()

    def set_indices(self, indices: List[TreeNode]):
        """
        Set new indices.
        :param indices: List of indices, usually Variables, Expressions, etc.
        :return: None.
        """
        for i, idx in zip(indices, range(len(indices))):
            self.add_edge(str(idx), i, n_type='index')

    def transpose(self, idx, include_data=True):
        '''
        Tranpose the multidimensional matrix.
        :param idx: New index order, comparable to tranpose of an ndarray.
        :param include_data: Also transpose the initial data?
        :return:  None.
        '''
        if include_data:
            self.get_node('var').init_data = self.get_node('var').init_data.transpose(idx)
        old_idxs = []
        for i in idx:
            old_idxs.append(self.get_node(str(idx[i])))
        for i in range(len(idx)):
            self.add_edge(str(i), old_idxs[idx[i]], n_type='index', replace=True)
        self.get_node('var').dim = [self.get_node('var').dim[idx[i]] for i in range(len(idx))]

    def __str__(self):
        """
        Get the string with Variable and indices.
        :return: The string.
        """
        s = str(self.get_node('var'))
        n = self.get_node_by_type('index')
        for i in n:
            s += '[' + str(i)
            if self.padding_to_offset:
                s += ' + ' + str(self.get_node('var').pads[n.index(i)][0])
            s += ']'
        return s
