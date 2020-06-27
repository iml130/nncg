from nncg.nodes.misc import Node


class QuantizedNode(Node):
    def __init__(self, node: Node):
        super().__init__()
        node.replace_self_with_path(self, self)
        self.add_edge('content', node)
