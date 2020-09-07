import os
import sys

compiler = 'g++ -mssse3 -std=c++11 -g -march=bonnell -DCNN_TEST '
compiler_O3 = 'g++ -O3 -mssse3 -std=c++11 -march=bonnell -DCNN_TEST '
compiler_check = 'g++ --version >/dev/null 2>/dev/null'

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
    cmd = c + path + " -o " + path[:path.rfind('.')]
    if os.system(cmd) != 0:
        print("Error compiling file with command: ", cmd)
        sys.exit(3)
