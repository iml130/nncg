import sys
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D, Flatten, \
    Dropout, BatchNormalization, LeakyReLU, InputLayer, Dense

from .nodes.cnn import *
from .nodes.language import CHeaderNode, CFooterNode
from .nodes.misc import *
from .nodes.controlflow import *
from .nodes.controlflow import UnrolledOperation
from .nodes.arithmetic import MACNodeSSE3
from .traverse.actions.searchnode import SearchNodeByType
from .traverse.actions.writecaction import WriteCAction
from .traverse.tree import Edge
from .compilercmds import compile, compiler_check
from .tools import print_progress_bar
from .allocation import Allocation


class NNCG:
    """
    Core class of this compiler.
    """
    root_node: Edge = None
    test_nodes: List[TestNode] = []

    def __init__(self):
        """
        Init this class.
        """
        self.id = ""
        self.test_nodes = []
        self.testing = None
        self.model = None

    def keras_compile(self,
                      imdb,
                      model,
                      code_path,
                      identifier=None,
                      image_mean=0,
                      arch="general",
                      testing=-1,
                      weights_method='direct'):
        """
        Main function to run the code generation.
        :param imdb: Image database as list of numpy array.
        :param model: Keras model.
        :param code_path: Path for writing code.
        :param identifier: Identifier of C code file.
        :param image_mean: Mean to subtract from image.
        :param arch: Architecture of target device.
        :param testing: Do testing? -1: all, otherwise number of tests.
        :param weights_method: How to store the weights and bias.
        :return: None.
        """

        self.testing = testing

        if identifier is not None:
            path = code_path + "/cnn_" + identifier + ".cpp"
        else:
            path = code_path

        exe_return_filename = "result.txt"

        self.model = model
        input_shape = model.layers[0].input.shape[1:].as_list()

        self.root_node = Edge('root', CHeaderNode(identifier, input_shape, weights_method), None, 'forward')

        cur_node = MeanNode(image_mean, self.root_node.target)
        cur_node = self.add_test_node(cur_node, None)

        # Read the Keras model layer by layer and add it to the graph

        for i, layer in enumerate(model.layers):
            print_progress_bar(i, len(model.layers), prefix='Generating code')
            if type(layer) == Convolution2D:
                cur_node = self.add_conv2d(layer, cur_node)
            elif type(layer) == MaxPooling2D:
                cur_node = self.add_maxpool2d(layer, cur_node)
            elif type(layer) == LeakyReLU:
                pass
            elif type(layer) == Dense:
                cur_node = self.add_dense(layer, cur_node)
                pass
            elif type(layer) == Flatten:
                cur_node = self.add_flatten(cur_node)
            elif type(layer) == Dropout:
                pass
            elif type(layer) == InputLayer:
                pass
            elif type(layer) == BatchNormalization:
                print("Warning: BatchNormalization not implemented")
            else:
                print("Unknown layer")
                sys.exit(1)

        CFooterNode(exe_return_filename, weights_method, cur_node)

        # Read in finished, lower to nodes that can be expressed in C.
        self.abstract_to_c()

        # Now convert the graph to the desired architecture. This will heavily change in future
        # when more architectures are supported.
        if arch == 'general':
            pass
        elif arch == 'sse3':
            self.to_sse3()

        print("")
        print("Writing...")

        self.write_c(path)

        print("Checking compiler version ...")
        if os.system(compiler_check) == 1:
            print("Compiler not found, not checking code file --> Finished.")
            sys.exit(0)
        if testing == 0:
            print("Finished")
            sys.exit(0)
        if testing == -1:
            testing = len(imdb)
        print("Compiling...")

        compile(path, optimize=False)

        print("Generation successful. Testing...")
        tested = 0
        fail = 0
        for im in np.random.permutation(imdb):
            if tested > testing:
                print("\nTest finished.")
                break

            im.astype("float32").tofile("img.bin")
            im = im.reshape(1, *im.shape)

            if os.name == 'nt':
                res = os.system(path[:path.rfind('.')])
            else:
                res = os.system("./" + path[:path.rfind('.')]) >> 8

            assert res == 0

            for n in self.test_nodes:
                n.test(im)
            tested += 1
            print_progress_bar(tested, testing,
                               suffix=", " + str((tested - fail) / tested * 100) + "% ok",
                               prefix='Evaluating')
        Allocation.reset()
        CHeaderNode.instance().reset()

    def to_sse3(self):
        """
        From a general architecture to SSE3.
        :return: None
        """
        desired_unroll = 4
        node_type = MACNode
        self.join_loops(desired_unroll, node_type)
        action = SearchNodeByType(node_type)
        self.root_node.traverse(action)
        for r in action.result:
            loop_to_unroll = r[-2]
            loop_to_unroll.unroll(desired_unroll)

        action = SearchNodeByType(UnrolledOperation)
        self.root_node.traverse(action)
        CHeaderNode.instance().intel_intr_required = True
        for r in action.result:
            u: UnrolledOperation = r[-1]
            node = u.get_node('content')
            if type(node is MACNode):
                if MACNodeSSE3.applicable(node):
                    MACNodeSSE3.apply(node)

    def join_loops(self, desired_unroll, node_type):
        """
        Join all loops in global graph required to be able to unroll them as desired, e.g. if you want to have
        MAC nodes 4 times in a tow to be able to replace them by a SIMD instruction.
        :param desired_unroll: How many times the inner loop should be unrollable. Join loops to reach this minimum.
        :param node_type: Join only loops containing this type of node.
        :return: None.
        """
        action = SearchNodeByType(node_type)
        self.root_node.traverse(action)
        mac_instances = []
        for r in action.result:
            r.pop()
            cur_list = []
            mac_instances.append(cur_list)
            for _node in reversed(r):
                if type(_node) is not LoopNode:
                    break
                cur_list.append(_node)
        for mi in mac_instances:
            depth = 0
            loops_to_join = []
            for _mac in mi:
                depth += _mac.stop
                loops_to_join.append(_mac)
                if depth > desired_unroll:
                    break
            mac_instances[mac_instances.index(mi)] = loops_to_join
        for i in mac_instances:
            root_loop = i[-1]
            while type(root_loop.get_node('content')) is LoopNode:
                root_loop = root_loop.deep_join()

    def add_conv2d(self, layer: Convolution2D, prev_node) -> Node:
        """
        Add a Conv2D node to global graph. A layer for testing is also added.
        :param layer: The Keras Conv2D layer.
        :param prev_node: Previous node.
        :return: The NNCG Conv2D node.
        """
        w = K.eval(layer.weights[0])
        b = K.eval(layer.bias)
        strides = layer.strides
        padding = layer.padding
        activation = layer.activation
        cur_node = Conv2DNode(w, b, strides, padding, prev_node)
        cur_node = self.add_activation(activation, cur_node)
        if self.testing != 0:
            cur_node = self.add_test_node(cur_node, layer)
        return cur_node

    def write_c(self, path):
        """
        Write the global graph as C code.
        :param path: File to write to.
        :return: None.
        """
        a = WriteCAction(path)
        self.root_node.traverse(a)

    def add_activation(self, activation: str, prev_node) -> Node:
        """
        Add an activation layer to global graph.
        :param activation: the layer type as string.
        :param prev_node: The previous node.
        :return: The added NNCG node.
        """
        if activation.__name__ == 'relu':
            prev_node = self.add_leaky_relu(0, prev_node)
        elif activation.__name__ == 'softmax':
            prev_node = self.add_softmax(prev_node)
        return prev_node

    @staticmethod
    def add_softmax(prev_node) -> SoftmaxNode:
        """
        Add a softmax layer.
        :param prev_node: Previous node.
        :return: The NNCG SoftmaxNode.
        """
        return SoftmaxNode(prev_node)

    @staticmethod
    def add_flatten(prev_node) -> FlattenNode:
        """
        Add a flatten layer. Note that this layer does not support all features of a Keras Flatten.
        :param prev_node:
        :return: The NNCG FlattenNode.
        """
        return FlattenNode(prev_node)

    def add_maxpool2d(self, layer: MaxPooling2D, prev_node) -> MaxPoolingNode:
        """
        Add a max pooling layer to the global graph.
        :param layer: The Keras MaxPooling2D layer.
        :param prev_node: The previous node.
        :return: The NNCG MaxPoolingNode.
        """
        size = layer.pool_size
        stride = layer.strides
        cur_node = MaxPoolingNode(size, stride, prev_node)
        if self.testing != 0:
            cur_node = self.add_test_node(cur_node, layer)
        return cur_node

    def add_dense(self, layer: Dense, prev_node) -> Node:
        """
        Add a Dense layer to the global graph. A layer for testing is also added.
        :param layer: The Keras Dense layer.
        :param prev_node: The previous node.
        :return: The NNCG DenseNode.
        """
        w = K.eval(layer.weights[0])
        b = K.eval(layer.bias)
        activation = layer.activation
        cur_node = DenseNode(w, b, prev_node)
        cur_node = self.add_activation(activation, cur_node)
        if self.testing != 0:
            cur_node = self.add_test_node(cur_node, layer)
        return cur_node

    @staticmethod
    def add_leaky_relu(alpha, prev_node):
        """
        Add a ReLU layer which is optional leaky.
        :param alpha: The leakyness parameter. Set this to 0 for a conventional ReLU.
        :param prev_node: The previous node.
        :return: The NNCG LeakyReLUNode.
        """
        return LeakyReLUNode(alpha, prev_node)

    def add_test_node(self, prev_node, layer):
        """
        Add a test node to global graph. Usually automatically used in other node adding functions.
        :param prev_node: The previous node.
        :param layer: The Keras layer to be compared with the NNCG feature layer at this point..
        :return: The NNCG TestNode.
        """
        if layer is None:
            func = None
            name = 'input'
        else:
            func = K.function([self.model.input, K.learning_phase()], [layer.output])
            name = layer.name
        n = TestNode(prev_node, func, name)
        self.test_nodes.append(n)
        return n

    def abstract_to_c(self):
        """
        Lower the global graph to nodes that can be expressed in C. Can only be applied after adding the Keras nodes
        to the graph and only once.
        :return: None.
        """
        n = self.root_node.target
        node_count = 0
        while n.has_edge('next'):  # Future: Use search for this
            node_count += 1
            n = n.get_node('next')

        n = self.root_node.target
        i = 0
        cont = True
        while cont:
            lowering = getattr(n, "lowering", None)
            if n.has_edge('next'):
                n = n.get_node('next')
            else:
                cont = False
            if callable(lowering):
                lowering()
            print_progress_bar(i, node_count, prefix='Lowering to C', suffix=str(n))
            i += 1
