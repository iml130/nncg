import numpy as np

def _len(a):
    """
    Giving the length with at least 1d.
    :param a:
    :return:
    """
    return len(np.atleast_1d(a))


def quantize_scale(min, max, type):
    """
    For future quantization.
    """
    if abs(min) > abs(max):
        v = abs(min)
    else:
        v = abs(max)
    if type == 'int8':
        return v / 127
    elif type == 'uint8':
        return v / 256


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='█'):
    """
    Call in a loop to create terminal progress bar.
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()
