import os
import sys

compiler = 'g++ -mssse3 -std=c++11 -g -march=bonnell -DCNN_TEST '
compiler_O3 = 'g++ -O3 -mssse3 -std=c++11 -march=bonnell -DCNN_TEST '
compiler_check = 'g++ --version'

# if os.name == 'nt':
#    compiler = 'wsl clang++ -mssse3 -std=c++11 -g -march=bonnell -DCNN_TEST '
#    compiler_O3 = 'wsl clang++ -O3 -mssse3 -std=c++11 -march=bonnell -DCNN_TEST '


def compile(path, optimize=False):
    """
    Compile the C file.
    :param path: Path to C file.
    :param optimize: Run compiler with optimizations.
    :return:
    """
    c = compiler_O3 if optimize else compiler
    if os.system(c + path + " -o " + path[:path.rfind('.')]) != 0:
        print("Error compiling file.")
        sys.exit(3)
