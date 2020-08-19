from __future__ import annotations
from typing import Dict
from copy import copy
from nncg.nodes.expressions import Variable


class Allocation:
    """
    Singleton class to allocate new variables.
    """
    allocations: Dict[str, Variable] = {}

    @staticmethod
    def reset():
        """
        Reset the current allocation. Must be done on a new NNCG run.
        :return:
        """
        Allocation.allocations = dict()

    @staticmethod
    def allocate_var(var_type, name, dim=None, alignment=0, init_data=None) -> Variable:
        """
        Allocate a variable.
        :param var_type: Type of data, a C type like "int" or "float".
        :param name: Name of the variable. A number will be added to get a unique name.
        :param dim: Dimension in case of an Array. Note that indices will only be written when used in an IndexVariable
                    instance.
        :param alignment: Required alignment, 0 for no alignment or e.g. 2 for a two-byte alignment.
        :param init_data: Initial data, e.g. trained weights.
        :return: The Variable.
        """
        index = 0
        if name in Allocation.allocations.keys():
            index = Allocation.allocations[name].index + 1
        Allocation.allocations[name] = Variable(var_type, name, dim, alignment, index, init_data)
        return copy(Allocation.allocations[name])
