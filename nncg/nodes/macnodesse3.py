from nncg.nodes.arithmetic import MACNode, Optimization
from nncg.nodes.controlflow import UnrolledOperation


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
        in a specific way to be replaceable by this implementation. Datatype must be float.
        :param other: The MACNode to be replaced.
        :return: True or False.
        """
        unrolled_op: UnrolledOperation = other.get_node('!content').get_node('!content')

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

        # Check data types

        if str(other.get_node('var1').get_type())[0:5] != 'float':
            return False

        if str(other.get_node('var1').get_type())[0:5] != 'float':
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