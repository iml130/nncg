from __future__ import annotations
from typing import List
import numpy as np
from nncg.allocation import Allocation
from nncg.nodes.expressions import Variable
from nncg.nodes.misc import Node
from nncg.writer import Writer


class CHeaderNode(Node):
    """
    Class for writing everything in a C file that is above the first CNN node, e.g. the MeanNode. Should
    be the root node of the complete graph.
    """
    __instance: CHeaderNode = None
    intel_intr_includes = '''
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
    '''

    math_include = '#include <math.h>\n'

    test_include = '''
#ifdef CNN_TEST
#include <stdio.h>
#endif

'''

    # Used to initialize the weights in case it is to large to put them into the C code file.
    weights_init_stdio = '''
#include <stdio.h>

void init_weight_float(float* w, int len, const char* name)
{{
    FILE *f = fopen(name, "rb");
    fread(w, sizeof(float), len, f);
    fclose(f);
}}

void init_weight_int8(int8_t* w, int len, const char* name)
{{
    FILE *f = fopen(name, "rb");
    fread(w, sizeof(int8_t), len, f);
    fclose(f);
}}

void init_weight_int16(int16_t* w, int len, const char* name)
{{
    FILE *f = fopen(name, "rb");
    fread(w, sizeof(int16_t), len, f);
    fclose(f);
}}

void init_weights()
{{
{}
}}

'''

    func_def = 'void cnn{id}(float {out_var_name}, float *{scores_var})\n{{\n'

    intel_intr_required = False
    math_required = False
    test_required = True

    out_var: Variable = None
    out_var_name = ''

    var_decls: List[Variable] = []
    pointer_decls: List[Variable] = []
    const_decls: List[Variable] = []

    def __init__(self, id, in_dim, weights_method):
        """
        Initialize the node.
        :param id: An identifier that is added to the function name, see func_def.
        :param in_dim: The three dimensional length of the input: H x W x C
        :param weights_method: The method how the weights are stored and initialized.
                               'direct': The weights are written into the C file.
                               'stdio': The weights are read using ANSI C stdio.
        """
        super().__init__()
        self.id = id
        self.in_dim = in_dim
        self.out_var = Allocation.allocate_var('float', 'x', in_dim)
        self.out_var.decl_written = True
        self.out_dim = in_dim
        self.weights_method = weights_method
        if weights_method == 'stdio':
            self.direct = False
            self.stdio = True
        elif weights_method == 'direct':
            self.direct = True
            self.stdio = False
        else:
            raise Exception('Unknown weights method.')
        CHeaderNode.__instance = self
        self.reset()

    @staticmethod
    def instance() -> CHeaderNode:
        """
        Gives the single instance of this class.
        :return: The instance.
        """
        return CHeaderNode.__instance

    def reset(self):
        """
        Clear all variables.
        :return:
        """
        self.var_decls = []
        self.pointer_decls = []
        self.const_decls = []

    def write_c(self):
        """
        Override the base write_c method. This method is called when the WriteC action writes the C code file.
        It is called before the child nodes are visited.
        :return: None.
        """
        self.snippet = ''
        weight_snippet = ''
        if self.id is None:
            self.id = ''

        # Remove the last added variable from any list for declarations, we will declare it in the
        # function declaration
        self.scores_var = CFooterNode.instance().in_var
        try:
            self.var_decls.remove(self.scores_var)
        except Exception:
            pass
        try:
            self.pointer_decls.remove(self.scores_var)
        except Exception:
            pass
        try:
            self.const_decls.remove(self.scores_var)
        except Exception:
            pass

        # Name and definition of the input variable for the C function definition. Variable is call 'out' because
        # it is not only the input but also the output of this layer and following nodes search for 'out_var'.
        self.out_var_name = str(self.out_var) + ''.join(['[' + str(i) + ']' for i in self.in_dim])

        # Add test code
        if self.test_required:
            self.snippet += self.test_include

        # Add include for math.h
        if self.math_required:
            self.snippet += self.math_include

        # Add header for Intel intrinsics
        if self.intel_intr_required:
            self.snippet += self.intel_intr_includes

        # Write all constants, primarily weights including the init data.
        for v in self.const_decls:
            self.snippet += v.get_def(self.direct).replace('{', '{{').replace('}', '}}')
            if self.stdio:
                # In this case the weights are later loaded from file.
                var_type = Variable.type_to_c(v.type)
                if var_type == 'float':
                    func_name='init_weight_float'
                elif var_type == 'int8_t':
                    func_name = 'init_weight_int8'
                elif var_type == 'int16_t':
                    func_name = 'init_weight_int16'
                else:
                    assert False
                weight_snippet += '\t{}(({}*){}, {}, "{}");\n'.format(func_name, var_type, str(v), np.prod(v.dim), str(v))
                Writer.write_data(v.init_data, str(v))
        if self.stdio:
            self.snippet += self.weights_init_stdio.format(weight_snippet).replace('{', '{{').replace('}', '}}')
        self.snippet += self.func_def
        # Now write all variable definitions. That are primarily the outputs of each layer.
        for v in self.var_decls:
            self.snippet += '\t' + v.get_def().replace('{', '{{').replace('}', '}}')
        # And now all pointer definitions.
        for v in self.pointer_decls:
            self.snippet += '\t' + v.get_pointer_decl().replace('{', '{{').replace('}', '}}')

        super().write_c()
        Writer.cur_depth += 1


class CFooterNode(Node):
    """
    Class for writing everything in a C file that is below the last CNN node. Should
    be the last node of the path of 'next' edges.
    """
    __instance: CFooterNode = None

    @staticmethod
    def instance() -> CFooterNode:
        """
        Gives the single instance of this class.
        :return: The instance.
        """
        return CFooterNode.__instance

    snippet = '''
    return;
}}

#ifdef CNN_TEST
#include <stdio.h>
#ifdef TIMING
#include <ctime>
#endif

int main()
{{
    int i, j, k, width, height, max_colour;
    unsigned char byte;
    float x[{x_dim}][{y_dim}][{z_dim}];
    float scores[{in_dim}];
    FILE *f = fopen("img.bin", "rb");
    fread((float*)x, sizeof(float), {x_dim} * {y_dim} * {z_dim}, f);
    fclose(f);
    {weights_init}

    cnn{id}(x, scores);
    FILE *w = fopen("{exe_return_filename}", "w");
    for (int i = 0; i < {in_dim}; i++)
        fprintf(w, "%f ", scores[i]);
    fclose(w);
}}
#endif
    '''
    end_c = '\n'

    def __init__(self, exe_return_filename, weights_method, prev_node):
        """
        Initialize the node.
        :param exe_return_filename: Name of file to write test results in.
        :param weights_method: The method how the weights are stored and initialized.
                               'direct': The weights are written into the C file.
                               'stdio': The weights are read using ANSI C stdio.
        :param prev_node: The previous node.
        """
        super().__init__(prev_node)
        CFooterNode.__instance = self
        dim = CHeaderNode.instance().in_dim
        id = CHeaderNode.instance().id
        self.in_dim = prev_node.out_dim
        self.in_var = prev_node.out_var
        self.x_dim = dim[0]
        self.y_dim = dim[1]
        if len(dim) > 2:
            self.z_dim = dim[2]
        else:
            self.z_dim = 1
        self.version = "5"
        if id is None:
            self.id = ''
        else:
            self.id = id
        self.exe_return_filename = exe_return_filename
        if weights_method == 'stdio':
            self.weights_init = 'init_weights();'
        elif weights_method == 'direct':
            self.weights_init = ''
        else:
            raise Exception('Unimplemented')

    def write_c(self):
        """
        Write the code of this node.
        :return: None.
        """
        Writer.cur_depth -= 1
        super().write_c()
