from __future__ import annotations
from nncg.nodes.expressions import *
from nncg.nodes.misc import Node


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

    def __init__(self, res_var: TreeNode, var1: TreeNode, prev_node: Node = None):
        """
        Init method for class. Initially there is not cast, cast is added by write_c()
        :param res_var: Assign to.
        :param var1: Assign from.
        :param prev_node: The previous node.
        """
        super().__init__(res_var, var1, prev_node)
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
    Base class for all arithmetic operations with three operands and a cast to change precision.
    res = (cast) var1 OP var2
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

        # Check if the res_var has a type with more precision and cast if
        # if var1.get_node('var').type == '
        res_var_w = Variable.type_to_width(res_var.get_type())
        var1_w = Variable.type_to_width(var1.get_type())
        var2_w = Variable.type_to_width(var2.get_type())
        if var1_w < res_var_w and var2_w < res_var_w:
            self.cast = '(' + Variable.type_to_c(res_var.get_type()) + ')'
        else:
            self.cast = ''

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
    Node for simple subtraction to change precision of calculation.
    res = var1 - var2
    """

    snippet = '{res_var} = {var1} - {var2};\n'


class AddNode(ThreeAddressNode):
    """
    Node for simple addition to change precision of calculation.
    res = var1 + var2
    """

    snippet = '{res_var} = {var1} + {var2};\n'


class MACNode(ThreeAddressNode):
    """
    Node for a multiply and accumulate.
    res += var1 * var2
    """

    snippet = '{res_var} += {cast} {var1} * {var2};\n'


class MultNode(ThreeAddressNode):
    """
    A node for multiplication.
    res = var1 * var2
    """
    snippet = '{res_var} = {cast} {var1} * {var2};\n'


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
