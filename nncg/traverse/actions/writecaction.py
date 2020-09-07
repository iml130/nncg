from nncg.traverse.traverseaction import TraverseAction
from nncg.writer import Writer


class WriteCAction(TraverseAction):
    """
    Action for writing C code. Only used once in main function.
    """

    def __init__(self, path):
        """
        Init the action.
        :param path:
        """
        super().__init__()
        self.traverse_edges = ['content', 'next']
        Writer.open(path)

    def _pre_action(self, edge) -> bool:
        """
        Just call write_c() of the current target. See overwritten method for details.
        :param edge: The currently visited Edge.
        :return: Always True.
        """
        edge.target.write_c()
        return True

    def _post_action(self, edge) -> bool:
        """
        LoopNodes and other nodes with 'content' nodes possibly raise the indentation and close brackets
        so these need a call when the content is left.
        :param edge:
        :return: Always True.
        """
        if edge.name_equal('content'):
            edge.owner.write_c_leave()
        return True

    def __del__(self):
        """
        If this instance is deleted we will not need the file anymore and it can be closed.
        :return: None.
        """
        Writer.close()
