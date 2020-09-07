from __future__ import annotations
from typing import List
import math
import numpy as np
from nncg.allocation import Allocation
from nncg.nodes.expressions import *
from nncg.nodes.misc import Node, AlternativesNode
from nncg.traverse.actions.replaceexpression import ReplaceExpression
from nncg.traverse.actions.deepcopy import DeepCopyLoop
from nncg.traverse.actions.searchnode import SearchNodeByName
from nncg.traverse.actions.deepcopy import DeepCopy
from nncg.writer import Writer


class LoopNode(Node):
    """
    Node representing an repeated execution of the 'next' path in 'content' edge. This is not an abstract node
    but is created by abstract nodes in the lowering process. The representation in C code is a for loop but
    it still contains all required information to unroll the loop for a vector operation like SSE.
    """
    content: Node

    snippet = 'for ({type} {var} = {start}; {var} < {stop}; {var} += {step}) {{\n'
    var: Variable

    @staticmethod
    def create_loops(dims: List[int]) -> (LoopNode, List[Variable]):
        """
        Automatically creates a nested loop structure for a given dimensions description. For each dimension a loop
        will be created such that all elements in a corresponding array will be accessed.
        :param dims: The dimensions of the array to be accessed.
        :return: The first loop and a list of all counter variables.
        """
        loops = [LoopNode(d) for d in np.atleast_1d(dims)]
        p = loops[0]
        for l in loops[1:]:
            p.add_edge('content', l)
            p = l
        return loops, [l.get_node('var') for l in loops]

    @staticmethod
    def create_loops_by_description(loops_descr: List[List[int]]) -> List[LoopNode]:
        """
        Automatically creates a nested loop structure according to the description.
        :param loops_descr: Description of loop structure in the format [[start, stop, step], [start, stop, step], ...].
        :return: The first LoopNode.
        """
        loops = []
        for ld in loops_descr:
            l = LoopNode(start=ld[0], stop=ld[1], step=ld[2])
            if len(loops) > 0:
                loops[-1].add_edge('content', l)
            loops.append(l)
        return loops

    @staticmethod
    def __xchange_temp_values(loop_vars: List[Variable], temporal_values) -> List:
        """
        Internal function to set the temporal values for the counter variables in the given loops. By setting these
        values the indices of IndexVariables used inside the loops can be calculated.
        :param loop_vars: List of variables to get the temporal values.
        :param temporal_values: List of temporal values. Must have the same dimensions as loop_vars.
        :return: The previous temporal values of the variables. Contains None if a variable did not have a temporal value.
        """
        old_vals = []
        for var, val in zip(loop_vars, temporal_values):
            old_vals.append(var.temporal_value)
            var.temporal_value = val
        return old_vals

    @staticmethod
    def get_access_pattern(loops: List[LoopNode], var: IndexedVariable):
        """
        Calculate which offset to the base address of an IndexVariable is accessed if the parent LoopsNode are
        incremented. This is a generator.
        :param loops: List of LoopNodes.
        :param var: The IndexVariable.
        :return: Yields a single offset.
        """
        loop_vars: List[Variable] = []
        temporal_values = []
        ignore = []

        # First, collect all variables and setup a list with all initial values.
        for l in loops:
            v = l.get_node('var')
            loop_vars.append(v)
            temporal_values.append(l.start)

        # Collect all count variables in the loops that are not used in the IndexVariable. These can be ignored.
        for v in loop_vars:
            action = SearchNodeByName(str(v))
            var.traverse(action)
            if action.result == []:
                ignore.append(v)

        stop = False
        while not stop:
            # Set temporal values in the loop counter vars.
            LoopNode.__xchange_temp_values(loop_vars, temporal_values)

            idx = 0
            access = 0
            dims = list(np.atleast_1d(var.get_node('var').dim))
            dims.append(1)

            # Now get all indices of the IndexedVariable and calculate the resulting address based on the
            # previously set temporal values.
            while var.has_edge(str(idx)):
                d = 1
                for i in range(idx + 1, len(dims)):
                    d *= dims[i]
                # get_node will give the idx'th index of the IndexedVariables. As we have set temporal values
                # str() will get a string not with variable names but with the temporal values so eval() can
                # calculate the given expression.
                access += math.floor(eval(str(var.get_node(str(idx))))) * d
                idx += 1

            # Now update the counter variables
            for i in reversed(range(len(loop_vars))):
                lvar = loop_vars[i]
                if lvar in ignore:
                    continue
                lvar.temporal_value += loops[i].step
                # If we are below the limit we are finished
                if lvar.temporal_value < loops[i].stop:
                    break
                else:
                    # If not, we have restart this loop and count up the next one.
                    lvar.temporal_value = loops[i].start
                    if i == 0:
                        stop = True

            # Change back the temporal values so they give the normal string with variables.
            temporal_values = LoopNode.__xchange_temp_values(loop_vars, len(loop_vars) * [None])

            yield access

    def __init__(self, stop, content=None, start=0, step=1, var_name='i'):
        """
        Initialize the LoopNode.
        :param stop: Upper limit of loop.
        :param content: The first Node of a 'next' path to be executed within the loop.
        :param start: Initial value.
        :param step: Step size each iteration,
        :param var_name: Optional variable name.
        """
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step
        if content is not None:
            self.add_edge('content', content)
        self.type = 'int'
        var = Allocation.allocate_var(self.type, var_name, [])
        self.add_edge('var', var)

    def unroll(self, times):
        """
        Unroll this loop. This will change the step size of this loop and it puts an UnrolledOperation between
        this LoopNode and its content the store meta information about this unroll. The content is then added to
        the of the original content <times> times.
        :param times: Size of unroll.
        :return: None.
        """
        assert self.step == 1  # Only this is supported for now
        r = (self.stop - self.start) % times
        if r != 0:
            self.split(self.stop - r)
        if times == self.stop - self.start:
            pass  # Here we could remove the loop, a lot of work for what?
        self.step = times
        # The unroll is not done by this function. Instead, to keep some meta information, an UnrolledOperation node
        # is added and this node will also do the unroll.
        UnrolledOperation.unroll_from_loop(self, times)

    def split(self, pos):
        """
        Split this LoopNode into two nodes. The second LoopNode will start at pos where the first new LoopNode
        ends.
        :param pos: Split the loop node at this counter value (new stop for first and new start for second).
        :return:
        """
        assert self.start < pos < self.stop
        scnd = DeepCopyLoop.deep_copy(self)
        if self.has_edge('next'):
            self.edges['next'].insert_node(scnd)
        else:
            self.add_edge('next', scnd)
        self.stop = pos
        scnd.start = pos

    def deep_join(self):
        """
        Joins the LoopNode which is the content of this node (-> nested LoopNode) into this LoopNode. Useful if
        the desired unroll is not possible because of an upper limit (stop) that is lower.
        :return: None.
        """
        next_loop = self.get_node('content')
        assert type(next_loop) is LoopNode
        # Joining means two counter variables will be replaced by one. We thus have to change all indices of
        # affected IndexVariables.
        linear_eq1 = (self.stop - self.start) / self.step
        linear_eq2 = (next_loop.stop - next_loop.start) / next_loop.step
        loop1_stop = round(linear_eq1)
        loop2_stop = round(linear_eq2)
        new_loop = LoopNode(loop1_stop * loop2_stop)
        old_idx_var1 = self.get_node('var')
        old_idx_var2 = next_loop.get_node('var')
        replace_1 = '{count_var}'
        if linear_eq2 != 0:
            replace_1 += ' / ' + str(int(linear_eq2))
        if self.step > 1:
            replace_1 += ' * ' + str(int(self.step))
        if self.start != 0:
            replace_1 += ' + ' + str(self.start)
        replace_2 = '({count_var}'
        if next_loop.step != 0:
            replace_2 += ' * ' + str(next_loop.step) + ')'
        else:
            replace_2 += ')'
        if linear_eq2 != 0:
            replace_2 += ' % ' + str(int(linear_eq2))
        else:
            replace_2 += ')'
        if next_loop.start != 0:
            replace_2 += ' + ' + str(next_loop.start)
        new_idx_var1: Expression = Expression(replace_1, count_var=new_loop.get_node('var'))
        new_idx_var2: Expression = Expression(replace_2, count_var=new_loop.get_node('var'))
        replace_action = ReplaceExpression(old_idx_var1, new_idx_var1)
        replace_action_2 = ReplaceExpression(old_idx_var2, new_idx_var2)
        self.traverse(replace_action)
        self.traverse(replace_action_2)
        self.merge(next_loop, replace=True)
        del self.edges['var']
        self.replace_self_with_path(new_loop, new_loop)
        return new_loop

    def write_c(self):
        """
        Override the base write_c method. This method is called when the WriteC action writes the C code file.
        It is called before the child nodes are visited. Must be overwritten here to handle indentation and write
        the content before next.
        :return: None.
        """
        _exp = self.snippet.format(**self.__dict__, **self.edges)
        Writer.write_c(_exp)
        Writer.cur_depth += 1

    def write_c_leave(self):
        """
        LoopNodes and other nodes with 'content' nodes possibly raise the indentation and close brackets so these
        need a call when the content is left. This is the case here.
        :return: None.
        """
        Writer.cur_depth -= 1
        Writer.write_c('}\n')

    def get_deep_length(self):
        """
        How many nested loops follow?
        :return: The number of further nested loops.
        """
        dlen = self.stop - self.start
        if type(self.content) is LoopNode and self.content.next_node is None:
            return dlen + self.content.get_deep_length()
        return dlen


class UnrolledOperation(AlternativesNode):
    """
    Abstract node to store meta information about an unroll operation that was performed at the point where
    this Node can be found.
    """

    def __init__(self, loop: LoopNode):
        """
        Initialize the node. This does not add it to the graph.
        :param loop: The loop to be unrolled.
        """
        super().__init__(loop)
        self.orig_loop = loop
        self.var = loop.get_node('var')

    def unroll(self, times):
        """
        Do the unroll. This does not add it to the graph.
        :param times: How many unrolls to do.
        :return: None
        """
        self.times = times
        original_content = DeepCopy.deep_copy(self.orig_loop)
        self.add_alternative(original_content)
        var_to_replace = self.var
        end_of_chain = self.orig_loop.get_node('content').search_path_end('next')
        # Every time we add the operations for a single unroll our offset to the original variable increases.
        for offset in range(1, times):
            var_w_offset = Expression('({var} + ' + str(offset) + ')', var=var_to_replace)
            replace_action = ReplaceExpression(var_to_replace, var_w_offset)
            nodes_copy = DeepCopy.deep_copy(original_content.get_node('content'))
            nodes_copy.traverse(replace_action)
            end_of_chain.add_edge('next', nodes_copy)
            end_of_chain = end_of_chain.search_path_end('next')

    @staticmethod
    def unroll_from_loop(loop, times):
        """
        Static method to get an unroll operation from a loop. It also replaces the loop with the UnrolledOperation.
        :param loop: The loop to be unrolled.
        :param times: How many unrolls.
        :return: None.
        """
        unrolled_op = UnrolledOperation(loop)
        unrolled_op.unroll(times)

    def get_access_pattern(self, pattern_len) -> Optional[Dict[IndexedVariable, List[int]]]:
        """
        Calculates the access pattern for the content of this UnrolledLoop.
        :param pattern_len: How many accesses you want to see per IndexedVariable?
        :return: The pattern. It's a dictionary where the IndexedVariable are the keys with a list of offsets as data.
        """
        n = self.get_node('content')
        loops = [n]
        while n.has_edge('!content'):
            n = n.get_node('!content')
            if type(n) is UnrolledOperation:
                pass
            elif type(n) is not LoopNode:
                return None
            else:
                loops.append(n)
        loops = list(reversed(loops))

        _n = self.get_node('content').get_node('content')
        pattern = {}

        # Assuming that two/three address nodes are not mixed as content
        while True:
            idx_vars = _n.get_vars()
            for v in idx_vars:
                if pattern.get(v) is None:
                    pattern[v] = []
                pattern[v].extend([p for _, p in zip(range(pattern_len), LoopNode.get_access_pattern(loops, v))])
            if _n.has_edge('next'):
                _n = _n.get_node('next')
            else:
                break

        return pattern

    def get_all_vars(self, name) -> List[IndexedVariable]:
        """
        Get all variables that can be accessed in the content of this node by the given name. E.g. get
        all 'res_var' Variables.
        :param name: The name of the edge to get the Variable.
        :return: List of Variables.
        """
        _n = self.get_node('content').get_node('content')
        _vars = []
        while True:
            _vars.append(_n.get_node(name))
            if _n.has_edge('next'):
                _n = _n.get_node('next')
            else:
                break
        return _vars
