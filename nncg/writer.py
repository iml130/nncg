from __future__ import annotations
import numpy as np


class Writer:
    """
    Singleton class to write the C code to a file. Expect indentation this class just writes
    and thus expects the final C code to be written.
    """

    f = None
    cur_depth = 0

    @staticmethod
    def write_c(_str: str):
        """
        Actually write C code to a file. File must be opened and closed.
        :param _str: The string to write.
        :return: None.
        """
        if Writer.f is None:
            return
        tabs = Writer.cur_depth * '\t'
        _str = _str.replace('\n', '\n' + tabs)
        if _str != '':
            while _str[-1] == '\t':
                _str = _str[:-1]
            while _str.find('\t#') != -1:
                _str = _str.replace('\t#', '#')
            if _str[0].strip() != '#':
                _str = tabs + _str
            Writer.f.write(_str)

    @staticmethod
    def write_data(d: np.ndarray, name):
        """
        Write float32 data to a file, e.g. weights.
        This writes directly to the file, no need to open something.
        :param d: Data as ndarray.
        :param name: Name of the file.
        :return: None.
        """
        d.tofile(name)

    @staticmethod
    def open(path):
        """
        Open the file for writing C code.
        :param path: Path to the file.
        :return: None.
        """
        if Writer.f is None:
            Writer.f = open(path, 'w')

    @staticmethod
    def close():
        """
        Close the C code file.
        :return: None
        """
        if Writer.f is not None:
            Writer.f.close()
            Writer.f = None
