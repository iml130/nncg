from nncg.nodes.arithmetic import *
from nncg.nodes.misc import *
from nncg.nodes.controlflow import LoopNode
from nncg.allocation import Allocation
from nncg.tools import _len
from nncg.quantization import QuantizedNode


class Conv2DNode(Node):
    """
    A Node representing a two-dimensional convolution. This is an abstract node to represent the meta
    information given by the Keras Conv2D. It is thus in HWC format. It must be lowered to be writeable
    as C code but should not be removed from graph to provide the meta information.
    """
    quantized = False
    in_var: Variable
    out_var: Variable
    access_pattern: List[int]

    def __init__(self, w: np.ndarray, b: np.ndarray, stride: tuple, padding: str, prev_node):
        """
        Initialize the Conv2DNode.
        :param w: Weights. Shape must be: kernel height, kernel width, channels in, channels out (number of filter)
                  as NumPy ndarray. Thus the weight from a Keras Conv2D can be passed without prior conversion.
        :param b: Bias. NumPy ndarray with length "channels out"
        :param stride: Tuple of 2.
        :param padding: Like in TensorFlow 'same' or 'valid'
        :param prev_node: The previous node.
        """
        self.in_var = prev_node.out_var
        x = self.in_var
        assert self.in_var.dim[2] == w.shape[2]
        assert w.shape[3] == b.shape[0]
        super().__init__(prev_node)
        self.in_dim = prev_node.out_dim
        self.w = w
        self.b = b
        self.stride = stride
        self.padding = padding
        self.H, self.W, self.C_IN = x.dim
        self.KH, self.KW, _, self.C_OUT = w.shape
        self.SH, self.SW = stride

        if padding == 'valid':
            H_OUT = int(np.ceil((self.H - self.KH + 1) / self.SH))
            W_OUT = int(np.ceil((self.W - self.KW + 1) / self.SW))
            self.pad_top = self.pad_bottom = self.pad_left = self.pad_right = 0
        elif padding == 'same':
            H_OUT = int(np.ceil(float(self.H) / float(self.SH)))
            W_OUT = int(np.ceil(float(self.W) / float(self.SW)))
            self.pad_along_height = max((H_OUT - 1) * self.SH + self.KH - self.H, 0)
            self.pad_along_width = max((W_OUT - 1) * self.SW + self.KW - self.W, 0)
            self.pad_top = int(self.pad_along_height // 2)
            self.pad_bottom = int(self.pad_along_height - self.pad_top)
            self.pad_left = int(self.pad_along_width // 2)
            self.pad_right = int(self.pad_along_width - self.pad_left)
        else:
            raise Exception("Unknown padding.")
        self.in_var.change_padding([[self.pad_top, self.pad_bottom],
                                    [self.pad_left, self.pad_right],
                                    [0, 0]])
        self.out_dim = (H_OUT, W_OUT, self.C_OUT)
        self.out_var = Allocation.allocate_var('float', 'x', self.out_dim)

    def lowering(self):
        """
        Create the loops required to express this node in ANSI C code without SIMD and connect this node with
        the new nodes via 'content' edge. This loop will stay in graph to provide meta information.
        :return: None.
        """

        # Create loops for settings the bias.
        b_var = Allocation.allocate_var(self.b.dtype, 'b', self.b.shape, init_data=self.b)
        out_var_idx = IndexedVariable(self.out_var)
        b_var_idx = IndexedVariable(b_var)

        # Create the loops using a descriptor.
        bias_loop_descr = [
            [0, self.out_dim[0], 1],
            [0, self.out_dim[1], 1],
            [0, self.out_dim[2], 1]
        ]
        bias_loops = LoopNode.create_loops_by_description(bias_loop_descr)
        b_h_loop = bias_loops[0]
        b_w_loop = bias_loops[1]
        b_c_loop = bias_loops[2]

        set_bias = AssignmentNode(out_var_idx, b_var_idx)
        b_c_loop.add_edge('content', set_bias)
        out_var_idx.set_indices([b_h_loop.get_node('var'), b_w_loop.get_node('var'), b_c_loop.get_node('var')])
        b_var_idx.set_indices([b_c_loop.get_node('var')])

        # Create the loops for convolution, again with descriptors
        conv_loop_descr = [
            [0, self.out_dim[0] * self.SH, self.stride[0]],
            [0, self.out_dim[1] * self.SW, self.stride[1]],
            [0, self.KH, 1],
            [0, self.KW, 1],
            [0, self.C_IN, 1],
            [0, self.C_OUT, 1]
        ]
        conv_loops = LoopNode.create_loops_by_description(conv_loop_descr)
        h_loop = conv_loops[0]
        w_loop = conv_loops[1]
        kh_loop = conv_loops[2]
        kw_loop = conv_loops[3]
        c_in_loop = conv_loops[4]
        c_out_loop = conv_loops[5]

        b_h_loop.add_edge('next', h_loop)

        w_var = Allocation.allocate_var(self.w.dtype, 'w', self.w.shape, init_data=self.w)
        out_var_idx = IndexedVariable(self.out_var)
        in_var_idx = IndexedVariable(self.in_var, False)
        w_var_idx = IndexedVariable(w_var, False)

        # Indices of IndexedVariables must respect the stride
        exp1 = Expression('{var} / {stride0}',
                          var=h_loop.get_node('var'),
                          stride0=Constant(self.stride[0]))
        exp2 = Expression('{var} / {stride1}',
                          var=w_loop.get_node('var'),
                          stride1=Constant(self.stride[1]))
        # And access to the image start at the upper left corner. But we have to add the current offset of the filter.
        exp3 = Expression('{var1} + {var2}',
                          var1=h_loop.get_node('var'),
                          var2=kh_loop.get_node('var'))
        exp4 = Expression('{var1} + {var2}',
                          var1=w_loop.get_node('var'),
                          var2=kw_loop.get_node('var'))
        out_var_idx.set_indices([exp1, exp2, c_out_loop.get_node('var')])
        in_var_idx.set_indices([exp3, exp4, c_in_loop.get_node('var')])
        w_var_idx.set_indices(
            [kh_loop.get_node('var'), kw_loop.get_node('var'), c_in_loop.get_node('var'), c_out_loop.get_node('var')])
        mac_node = MACNode(out_var_idx, w_var_idx, in_var_idx)
        c_out_loop.add_edge('content', mac_node)

        # These variables must be declared (partially with initial data) at the beginning of the function
        self.var_decls.append(self.out_var)
        self.const_decls.append(w_var)
        self.const_decls.append(b_var)

        # Don't remove this node, just put everything as content to this node.
        self.add_edge('content', b_h_loop)

    def quantize(self, x_scale):
        """
        Quantize this node.
        :param x_scale: A factor previously determined by quantize_scale() for scaling the weights. Used for bias here.
        :return: None.
        """
        min = np.min([np.min(self.w), np.min(self.b)])
        max = np.max([np.max(self.w), np.max(self.b)])
        self.scale = QuantizedNode.quantize_scale(min, max, 'int8')
        self.w = (self.w / self.scale).astype('int8')
        self.b = (self.b / self.scale / x_scale).astype('int16')
        #self.out_var.type = 'int'


class LeakyReLUNode(Node):
    """
    A (leaky) ReLU node. It must be lowered to be writeable as C code but should not be
    removed from graph to provide the meta information.
    """

    def __init__(self, alpha, prev_node):
        """
        Initialize the LeakyReLUNode.
        :param alpha: The leakyness of this node. 0 for a non-leaky (normal) ReLU.
        :param prev_node:
        """
        super().__init__(prev_node)
        self.alpha = alpha
        self.in_var = prev_node.out_var
        self.in_dim = prev_node.out_dim
        self.out_dim = self.in_dim
        self.out_var = Allocation.allocate_var('float', 'x', self.out_dim)

    def lowering(self):
        """
        Create the loops required to express this node in ANSI C code without SIMD.
        This loop will stay in graph to provide meta information.
        :return: None.
        """
        loops, idxs = LoopNode.create_loops(self.in_var.dim)
        in_var_idx = IndexedVariable(self.in_var)
        out_var_idx = IndexedVariable(self.out_var)
        in_var_idx.set_indices(idxs)
        out_var_idx.set_indices(idxs)
        condition = Expression('{t_var_idx} < 0', t_var_idx=in_var_idx)
        if self.alpha == 0:
            false_exp = Constant(0)
        else:
            false_exp = Expression('{alpha} * {t_var_idx}', t_var_idx=in_var_idx)
        cond_node = ConditionalNode(out_var_idx, condition, false_exp, in_var_idx)
        loops[-1].add_edge('content', cond_node)
        self.var_decls.append(self.out_var)

        # Meta information of this node not required yet, so delete this node and replace it with the loops.
        self.add_edge('content', loops[0])


class DenseNode(Node):
    """
    A Dense node. It must be lowered to be writeable as C code but should not be
    removed from graph to provide the meta information.
    """
    
    def __init__(self, w, b, prev_node):
        """
        Initialize the DenseNode.
        :param w: The weights in two dimensions: channels in, channels out. It is compatible to Keras.
        :param b: The bias with one dimension: channels out. It is compatible to Keras.
        :param prev_node: The previous node.
        """
        super().__init__(prev_node)
        self.w = w
        self.b = b
        self.out_dim = w.shape[1]
        self.in_dim = prev_node.out_dim
        self.in_var = prev_node.out_var
        self.out_var = Allocation.allocate_var('float', 'x', self.out_dim)

    def lowering(self):
        """
        Create the loops required to express this node in ANSI C code without SIMD and replace this node.
        This loop will stay in graph to provide meta information.
        :return: None.
        """
        b_var = Allocation.allocate_var('float', 'b', self.b.shape, init_data=self.b)
        b_var_idx = IndexedVariable(b_var)

        # Make sure that e.g. Flatten has been applied before. In Keras it is not required but it makes
        # things easier.
        assert _len(self.in_dim) == 1

        # Assign bias to output variable
        out_var_idx = IndexedVariable(self.out_var)
        b_loop = LoopNode(self.out_dim)
        out_var_idx.set_indices([b_loop.get_node('var')])
        b_var_idx.set_indices([b_loop.get_node('var')])
        set_bias = AssignmentNode(out_var_idx, b_var_idx)
        b_loop.add_edge('content', set_bias)

        # Loops for multiplication
        out_var_idx = IndexedVariable(self.out_var)
        in_loop = LoopNode(self.in_dim)
        out_loop = LoopNode(self.out_dim)
        out_var_idx.set_indices([out_loop.get_node('var')])
        w_var = Allocation.allocate_var('float', 'w', self.w.shape, init_data=self.w)
        in_var_idx = IndexedVariable(self.in_var, False)
        w_var_idx = IndexedVariable(w_var, False)
        in_var_idx.set_indices([in_loop.get_node('var')])
        w_var_idx.set_indices([in_loop.get_node('var'), out_loop.get_node('var')])
        mac_node = MACNode(out_var_idx, in_var_idx, w_var_idx)

        b_loop.add_edge('next', in_loop)
        in_loop.add_edge('content', out_loop)
        out_loop.add_edge('content', mac_node)

        self.var_decls.append(self.out_var)
        self.const_decls.append(w_var)
        self.const_decls.append(b_var)

        # Meta data not required yet so remove this node
        self.add_edge('content', b_loop)


class FlattenNode(Node):
    """
    Node that lowers the dimension.
    """
    def __init__(self, prev_node):
        """
        Initialize this node.
        :param prev_node: The previous node.
        """
        super().__init__(prev_node)
        self.in_dim = prev_node.out_dim
        self.out_dim = np.prod(self.in_dim)
        self.in_var = prev_node.out_var
        self.out_var = Allocation.allocate_var('float', 'x', self.out_dim)

    def lowering(self):
        """
        Create the loops required to express this node in ANSI C code without SIMD and replace this node.
        This loop will stay in graph to provide meta information.
        :return: None.
        """
        n = AssignmentNode(self.out_var, self.in_var)
        self.pointer_decls.append(self.out_var)
        self.add_edge('content', n)


class MaxPoolingNode(Node):
    """
    Node for max pooling.
    """
    def __init__(self, size, stride, prev_node):
        """
        Initialize the node.
        :param size: The size of the max filter with two dimensions: H x W. It is Keras compatible.
        :param stride: The stride with two dimensions: H x W. It is Keras compatible.
        :param prev_node: The previous Node.
        """
        super().__init__(prev_node)
        self.size = size
        self.stride = stride
        self.in_dim = prev_node.out_dim
        self.in_var = prev_node.out_var
        self.h_loop_end = self.in_dim[0] - size[0] + 1
        self.w_loop_end = self.in_dim[1] - size[1] + 1
        x_res = int(np.ceil(self.h_loop_end / stride[0]))
        y_res = int(np.ceil(self.w_loop_end / stride[1]))
        self.out_dim = (x_res, y_res, self.in_dim[2])
        self.out_var = Allocation.allocate_var('float', 'x', self.out_dim)

    def lowering(self):
        """
        Create the loops required to express this node in ANSI C code without SIMD and replace this node.
        This loop will stay in graph to provide meta information.
        :return: None.
        """
        h_loop = LoopNode(stop=self.h_loop_end,
                          step=self.stride[0])

        w_loop = LoopNode(stop=self.w_loop_end,
                          step=self.stride[1])
        h_loop.add_edge('content', w_loop)

        c_loop = LoopNode(self.in_dim[2])
        w_loop.add_edge('content', c_loop)

        exp1 = Expression('{var} / {stride0}', var=h_loop.get_node('var'), stride0=Constant(self.stride[0]))
        exp2 = Expression('{var} / {stride1}', var=w_loop.get_node('var'), stride1=Constant(self.stride[1]))
        out_var_idx = IndexedVariable(self.out_var)
        in_var_idx = IndexedVariable(self.in_var, False)
        out_var_idx.set_indices([exp1, exp2, c_loop.get_node('var')])
        in_var_idx.set_indices([h_loop.get_node('var'), w_loop.get_node('var'), c_loop.get_node('var')])

        init = AssignmentNode(out_var_idx, in_var_idx)
        c_loop.add_edge('content', init)

        kh_loop = LoopNode(self.size[0])
        init.add_edge('next', kh_loop)

        kw_loop = LoopNode(self.size[1])
        kh_loop.add_edge('content', kw_loop)

        exp3 = Expression('{var1} + {var2}', var1=h_loop.get_node('var'), var2=kh_loop.get_node('var'))
        exp4 = Expression('{var1} + {var2}', var1=w_loop.get_node('var'), var2=kw_loop.get_node('var'))
        out_var_idx = IndexedVariable(self.out_var)
        in_var_idx = IndexedVariable(self.in_var, False)

        out_var_idx.set_indices([exp1, exp2, c_loop.get_node('var')])
        in_var_idx.set_indices([exp3, exp4, c_loop.get_node('var')])

        condition = Expression('{var_in} > {var_out}', var_in=in_var_idx, var_out=out_var_idx)
        n = ConditionalNode(out_var_idx, condition, in_var_idx, out_var_idx)
        kw_loop.add_edge('content', n)
        self.add_edge('content', h_loop)
        self.var_decls.append(self.out_var)


class SoftmaxNode(Node):
    """
    Node for the softmax activation.
    """
    def __init__(self, prev_node):
        """
        Initialize the node.
        :param prev_node:
        """
        super().__init__(prev_node)
        self.in_dim = prev_node.out_dim
        if type(self.in_dim) is list:
            c = 0
            for d in self.in_dim:
                if d > 1:
                    c += 1
            assert c == 1
        self.out_dim = self.in_dim
        self.in_var = prev_node.out_var
        self.out_var = Allocation.allocate_var('float', 'x', self.out_dim)

    def lowering(self):
        """
        Create the loops required to express this node in ANSI C code without SIMD and replace this node.
        This loop will stay in graph to provide meta information.
        :return: None.
        """
        t_var = Allocation.allocate_var('float', 'flat_x', np.prod(self.out_dim))
        t_var_idx = IndexedVariable(t_var)
        n = AssignmentNode(t_var, self.in_var)
        sum_var = Allocation.allocate_var('float', 'sum', [])
        sum_loop = LoopNode(t_var.dim)
        sum_exp = Expression('{sum_var} += expf({t_var_idx});', sum_var=sum_var, t_var_idx=t_var_idx)
        sum_node = ExpressionNode(sum_exp)
        sum_loop.add_edge('content', sum_node)
        t_var_idx.set_indices([sum_loop.get_node('var')])
        out_var_idx = IndexedVariable(self.out_var)
        loops, idxs = LoopNode.create_loops(self.in_var.dim)
        out_var_idx.set_indices(idxs)
        in_var_idx = IndexedVariable(self.in_var)
        in_var_idx.set_indices(idxs)
        exp = Expression('{out_var_idx} = expf({in_var_idx}) / {sum_var};',
                         out_var_idx=out_var_idx, in_var_idx=in_var_idx, sum_var=sum_var)
        node = ExpressionNode(exp)
        loops[-1].add_edge('content', node)
        sum_loop.add_edge('next', loops[0])
        n.add_edge('next', sum_loop)
        self.pointer_decls.append(t_var)
        self.var_decls.append(self.out_var)
        self.var_decls.append(sum_var)
        self.math_required = True
        self.add_edge('content', n)


class MeanNode(Node):
    """
        A node to subtract the image mean before the CNN is executed. This kind of layer is not available in Keras
        but useful to do the preprocessing.
        out_var = in_var - mean
    """
    in_var: Variable
    out_var: Variable

    def __init__(self, mean, prev_node):
        """
        Initialize the node.
        :param mean: The mean to be subtracted as scalar.
        :param prev_node: The previous node.
        """
        super().__init__(prev_node)
        self.in_dim = prev_node.out_dim
        self.out_dim = self.in_dim
        self.in_var = prev_node.out_var
        self.out_var = Allocation.allocate_var('float', 'x', self.out_dim)
        self.mean = mean

    def lowering(self):
        """
        Create the loops required to express this node in ANSI C code without SIMD and replace this node.
        This loop will stay in graph to provide meta information.
        :return: None.
        """
        out_idx_var = IndexedVariable(self.out_var)
        in_idx_var = IndexedVariable(self.in_var)
        sub_node = SubNode(out_idx_var, in_idx_var, Constant(self.mean))
        n = sub_node
        count_vars = []
        for d in reversed(self.out_dim):
            n = LoopNode(d, n)
            count_vars.append(n.get_node('var'))
        count_vars = list(reversed(count_vars))
        in_idx_var.set_indices(count_vars)
        out_idx_var.set_indices(count_vars)
        # Meta data not required yet so remove this node
        self.add_edge('content', n)
        self.var_decls.append(self.out_var)
