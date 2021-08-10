from nncg.nodes.arithmetic import MACNode, Optimization, AssignmentNode, AddNode, Constant
from nncg.nodes.controlflow import UnrolledOperation
from nncg.nodes.misc import Node
from nncg.nodes.controlflow import LoopNode
from nncg.allocation import Allocation
from nncg.traverse.actions.searchnode import SearchNodeByType
from nncg.nodes.expressions import IndexedVariable, Expression
from nncg.nodes.cnn import Conv2DNode
from nncg.nodes.language import CHeaderNode
from nncg.nodes.funccall import FuncCallNode


class Int8SSE3Preprocessing(Node):
    """
    Node for preprocessing before using MACNodeInt8SSE3.
    """
    def __init__(self, H, W, C_OUT):
        """
        Init the Node. Immediately creates Nodes for writing C code as this node is applied after
        general lowering.
        :param H: Height.
        :param W: Width.
        :param C_OUT: Number of output channels.
        """
        super().__init__()
        loop_descr = [
            [0, H, 1],
            [0, W, 1],
            [0, C_OUT, 1]
        ]
        l = LoopNode.create_loops_by_description(loop_descr)
        self.add_edge('content', l[0])
        self.sse_var = Allocation.allocate_var('__m128i', 'cx', [H, W, C_OUT])
        sse_var_idx = IndexedVariable(self.sse_var)
        h_idx = l[0].get_node('var')
        w_idx = l[1].get_node('var')
        c_idx = l[2].get_node('var')
        sse_var_idx.set_indices([h_idx, w_idx, c_idx])
        an = AssignmentNode(sse_var_idx, Expression('_mm_setzero_si128()'))
        l[2].add_edge('content', an)
        self.var_decls.append(self.sse_var)


class Int8SSE3Postprocessing(Node):
    """
    Node for postprocessing after using MACNodeInt8SSE3. For internal use by MACNodeInt8SSE3.
    """
    def __init__(self, res_var, sse_var, H, W, C_OUT):
        """
        Init the Node. Immediately creates Nodes for writing C code as this node is applied after
        general lowering.
        :param res_var: The Variable that is the output of the original Node that was quantized.
        :param sse_var: The Variable for storing the intermediate quantized results.
        :param H: Output height.
        :param W: Output width.
        :param C_OUT: Channels out.
        """
        super().__init__()
        loop_descr = [
            [0, H, 1],
            [0, W, 1],
            [0, C_OUT, 1]
        ]
        l = LoopNode.create_loops_by_description(loop_descr)
        self.add_edge('content', l[0])
        sse_var_idx = IndexedVariable(sse_var)
        res_var_idx = IndexedVariable(res_var)
        h_idx = l[0].get_node('var')
        w_idx = l[1].get_node('var')
        c_idx = l[2].get_node('var')
        sse_var_idx.set_indices([h_idx, w_idx, c_idx])
        res_var_idx.set_indices([h_idx, w_idx, c_idx])
        lo_var = Allocation.allocate_var('__m128i', 'lo')
        l1 = AssignmentNode(lo_var, Expression('_mm_srai_epi32(_mm_unpacklo_epi16({qx}, {qx}), 16);', qx=sse_var_idx))
        hi_var = Allocation.allocate_var('__m128i', 'hi')
        l2 = AssignmentNode(hi_var, Expression('_mm_srai_epi32(_mm_unpackhi_epi16({qx}, {qx}), 16);', qx=sse_var_idx), l1)
        sum1_var = Allocation.allocate_var('__m128i', 'sum1')
        l3 = AssignmentNode(sum1_var, Expression('_mm_hadd_epi32({hi}, {lo});', lo=lo_var, hi=hi_var), l2)
        sum2_var = Allocation.allocate_var('__m128i', 'sum2')
        l4 = AssignmentNode(sum2_var, Expression('_mm_hadd_epi32({sum1}, {sum1});', sum1=sum1_var), l3)
        temp_var = Allocation.allocate_var('int', 'temp_res', [4])
        l5 = FuncCallNode(Expression('_mm_store_si128((__m128i*)&{res}, {sum2});', res=temp_var, sum2=sum2_var), l4)
        temp_var_idx_0 = IndexedVariable(temp_var)
        temp_var_idx_0.set_indices([Constant('0')])
        temp_var_idx_1 = IndexedVariable(temp_var)
        temp_var_idx_1.set_indices([Constant('1')])
        l6 = AddNode(res_var_idx, res_var_idx, temp_var_idx_0, l5)
        l7 = AddNode(res_var_idx, res_var_idx, temp_var_idx_1, l6)
        l[2].add_edge('content', l1)
        self.var_decls.append(lo_var)
        self.var_decls.append(hi_var)
        self.var_decls.append(sum1_var)
        self.var_decls.append(sum2_var)
        self.var_decls.append(temp_var)


class MACNodeInt8SSE3(MACNode, Optimization):
    """
    Node for a quad multiply and accumulate for SSE3 CPUs.
    """
    snippet = '''{{
    __m128i w, x, y;
    x = _mm_lddqu_si128((__m128i*)&{var2});
    w = _mm_lddqu_si128((__m128i*)&{var1});
    x = _mm_maddubs_epi16(x, w);
    {res_var} = _mm_adds_epi16(x, {res_var});
}}
'''

    @classmethod
    def applicable(cls, other):
        """
        Determine if this SSE3 with quantization implementation is applicable as replacement for a simple MACNode. The
        MACNode must be within an UnrolledOperation. The operands res_var and var1 must be accessed
        in a specific way to be replaceable by this implementation. Also datatype must be Int8.
        You can also apply this on a supported CNN node, e.g. Conv2DNode to check beforehand.
        :param other: The MACNode to be replaced or the CNN Node (e.g. Conv2DNode).
        :return: True or False.
        """

        if type(other) == Conv2DNode:
            # ToDo: First step, let it working without loop joining, probably could be possible to join
            #       KH, KW, C_IN
            #if other.C_IN * other.KH * other.KW < 16:
            #    return False
            if other.C_IN < 16:
                return False
            macnode = SearchNodeByType.get_next(other, MACNode, ['content', 'next'])
        elif type(other) == MACNode:
            unrolled_op: UnrolledOperation = other.get_node('!content').get_node('!content')

            # The UnrolledOperation must execute 16 MACNodes in a row that are then replaced.
            if unrolled_op.times != 16:
                return False
            pattern = unrolled_op.get_access_pattern(16)

            # Check  if the 16 MACs write to res_var in a row as this is done by the pattern above
            v = unrolled_op.get_all_vars('res_var')
            if [pattern[_v][0] for _v in v] != [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
                return False

            macnode = unrolled_op.get_node('content').get_node('content')

        # Check data types
        if macnode.get_node('var1').get_type() != 'int8':
            return False

        if macnode.get_node('var2').get_type() != 'uint8':
            return False

        return True

    @staticmethod
    def apply(root_node):
        """
        The root_node must be a MACNode, followed by further MACNodes, in an UnrolledOperation.
        It is then replaced by a single MACNodeInt8SSE3.
        :param root_node: Root node to be replaced.
        :return: None
        """
        n_sse3 = MACNodeInt8SSE3.from_threeaddress(root_node)
        unrolled_op = root_node.get_node('!content')
        unrolled_op.add_edge('content', n_sse3, replace=True)
        root_node.get_node('res_var').get_node('var').set_alignment(2)
        conv_node = SearchNodeByType.get_next(n_sse3, Conv2DNode, ['!content', '!next'])
        preproc_node = Int8SSE3Preprocessing(conv_node.H, conv_node.W, conv_node.C_OUT)
        n_sse3.get_node('res_var').edges['var'].set_target(preproc_node.sse_var)
        postproc_node = Int8SSE3Postprocessing(conv_node.out_var, preproc_node.sse_var, *conv_node.out_dim)
        conv_node.get_node('!next').edges['next'].insert_node(preproc_node)
        conv_node.edges['next'].insert_node(postproc_node)
        pass
