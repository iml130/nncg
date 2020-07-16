from __future__ import annotations
from nncg.nodes.expressions import *
from nncg.nodes.misc import Node
from nncg.nodes.controlflow import UnrolledOperation


class Optimization:
    """
    Base class for classes that for replacing parts of graph with optimized versions.
    """

    @classmethod
    def applicable(cls, other):
        """
        Determine if this optimization is applicable as replacement for other.
        :param other: TreeNode root to be replaced.
        :return: True or False.
        """
        pass

    @staticmethod
    def apply(root_node):
        """
        Apply the optimization
        :param root_node: Root node to be replaced, e.g. a MACNode to be replaced with MACNodeSSE3.
        :return: None
        """
        pass


class TwoAddressNode(Node):
    """
    Base class for all arithmetic operations with two operands.
    res OP var1
    """
    snippet = ''

    def __init__(self, res_var: TreeNode, var1: TreeNode, prev_node: Node = None):
        """
        Init method for class.
        :param res_var: First operand, usually the variable storing the result of this operation.
        :param var1: Second operand.
        :param prev_node: The previous node.
        """
        super().__init__(prev_node)
        self.add_edge('res_var', res_var, 'var')
        self.add_edge('var1', var1, 'var')

    @classmethod
    def from_twoaddress(cls, n: TwoAddressNode):
        """
        Create new instance of this class (or class that inherits from this class) from other TwoAddressNode.
        :param n: The other TwoAddressNode.
        :return: New instance.
        """
        return cls(*n.get_vars())

    def get_vars(self) -> List[IndexedVariable]:
        """
        Get both operands as list.
        :return: List of both operands.
        """
        return [self.get_node('res_var'), self.get_node('var1')]


class AssignmentNode(TwoAddressNode):
    """
    Just assign to a new Variable, usually a pointer. Useful for casting to different types, e.g.
    to flatten multidimensional arrays.
    res = var1
    """

    snippet = '{res_var} = {cast}{var1};\n'

    def __init__(self, res_var: TreeNode, var1: TreeNode):
        """
        Init method for class. Initially there is not cast, cast is added by write_c()
        :param res_var: Assign to.
        :param var1: Assign from.
        """
        super().__init__(res_var, var1)
        self.cast = ''

    def write_c(self):
        """
        Override the base write_c method. This method is called when the WriteC action writes the C code file.
        It is called before the child nodes are visited. Automatically adds the required cast.
        :return: None.
        """
        res_var: Variable = self.get_node('res_var')
        var1: Variable = self.get_node('var1')

        # Only adds a cast if both are Variables and not e.g. IndexedVariables and has different dimensions, i.e.
        # both should be C style arrays but without [] operator.
        if type(var1) == Variable and type(res_var) == Variable and \
                np.any(var1.dim != res_var.dim):
            self.cast = res_var.get_cast()
        super().write_c()


class ThreeAddressNode(Node):
    """
    Base class for all arithmetic operations with three operands.
    res = var1 OP var2
    """
    snippet = ''

    def __init__(self, res_var: TreeNode, var1: TreeNode, var2: TreeNode, prev_node: Node = None):
        """
        Init method for class.
        :param res_var: First operand, usually the variable storing the result of this operation.
        :param var1: Second operand.
        :param var2: Third operand.
        :param prev_node: The previous node.
        """
        super().__init__(prev_node)
        self.add_edge('res_var', res_var, 'var')
        self.add_edge('var1', var1, 'var')
        self.add_edge('var2', var2, 'var')

    @classmethod
    def from_threeaddress(cls, n: ThreeAddressNode):
        """
        Create new instance of this class (or class that inherits from this class) from other ThreeAddressNode.
        :param n: The other ThreeAddressNode.
        :return: New instance.
        """
        return cls(*n.get_vars())

    def get_vars(self) -> List[IndexedVariable]:
        """
        Get all operands as list.
        :return: List of both operands.
        """
        return [self.get_node('res_var'), self.get_node('var1'), self.get_node('var2')]


class SubNode(ThreeAddressNode):
    """
    Node for simple subtraction.
    res = var1 - var2
    """

    snippet = '{res_var} = {var1} - {var2};\n'


class MACNode(ThreeAddressNode):
    """
    Node for a multiply and accumulate.
    res += var1 * var2
    """

    snippet = '{res_var} += {var1} * {var2};\n'


class MultNode(ThreeAddressNode):
    snippet = '{res_var} = {var1} * {var2};\n'


class MACNodeSSE3(MACNode, Optimization):
    """
    Node for a quad multiply and accumulate for SSE3 CPUs.
    """
    snippet = '''{{
    __m128 w, x, y;
    w = _mm_load_ps((float*)&{var1});
    x = _mm_load_ps1(&{var2});
    y = _mm_mul_ps(w, x);
    x = _mm_load_ps((float*)&{res_var});
    x = _mm_add_ps(x, y);
    _mm_store_ps((float*)&{res_var}, x);
}}
'''

    @classmethod
    def applicable(cls, other: MACNode):
        """
        Determine if this SSE3 implementation is applicable as replacement for a simple MACNode. The
        MACNode must be within an UnrolledOperation. The operands res_var and var1 must be accessed
        in a specific way to be replaceable by this implementation.
        :param other: The MACNode to be replaced.
        :return: True or False.
        """
        unrolled_op: UnrolledOperation = other.get_node('!content')

        # The UnrolledOperation must execute 4 MACNodes in a row that are then replaced.
        if unrolled_op.times != 4:
            return False
        pattern = unrolled_op.get_access_pattern(4)

        # Check  if the 4 MACs write to res_var in a row as this is done by the pattern above
        v = unrolled_op.get_all_vars('res_var')
        if [pattern[_v][0] for _v in v] != [0, 1, 2, 3]:
            return False

        # Now check if every access to var1 is aligned which is required by _mm_load_ps
        v = unrolled_op.get_all_vars('var1')
        if [_v % 4 for _v in pattern[v[0]]] != [0, 0, 0, 0]:
            return False

        return True

    @staticmethod
    def apply(root_node):
        """
        The root_node must be a MACNode, followed by three further MACNode, in an UnrolledOperation.
        It is then replaced by a single MACNodeSSE3.
        :param root_node: Root node to be replaced.
        :return: None
        """
        n_sse3 = MACNodeSSE3.from_threeaddress(root_node)
        unrolled_op = root_node.get_node('!content')
        unrolled_op.add_edge('content', n_sse3, replace=True)
        root_node.get_node('res_var').get_node('var').set_alignment(2)


class ConditionalNode(Node):
    """
    A single operation that checks if a condition is true and executes the one or the other expression.
    res = condition ? true_var : false_var
    """

    snippet = '{res_var} = {condition} ? {true_var} : {false_var};\n'

    def __init__(self, res_var: Variable, condition: Expression, true_var: Expression, false_var: Expression,
                 prev_node: Node = None):
        """
        Initializes the ConditionalNode
        :param res_var: The Variable storing the result.
        :param condition: The condition to be checked.
        :param true_var: Expression executed if condition is true.
        :param false_var: Expression executed if condition is false.
        :param prev_node: The previous node.
        """
        super().__init__(prev_node)
        self.add_edge('res_var', res_var, 'var')
        self.add_edge('true_var', true_var, 'var')
        self.add_edge('false_var', false_var, 'var')
        self.add_edge('condition', condition, 'expression')
