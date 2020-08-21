from glob import glob
import pickle
import random
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical


def load_imdb(imgdb_path):
    """
    Load an image database written in custom format.
    :param imgdb_path: Path the the base file. Second file will be the path with appended ".x".
    :return: The image database.
    """
    images = {}
    with open(imgdb_path, "rb") as f:
        images["mean"] = pickle.load(f)
        images["y"] = pickle.load(f)
    with open(imgdb_path + '.x', "rb") as f:
        images["images"] = np.load(f)
    return images


def random_imdb(num_imgs, x_dim):
    """
    Generate a data structure with the layout of the image database but filled with random data.
    :param num_imgs: Number of random images.
    :param x_dim: Dimension of the random images as list with three entries.
    :return: Randomly filled image database structure.
    """
    return [np.random.rand(*x_dim) for _ in range(num_imgs)]


def load_images(path, res, clss, clss_num, db, rotate=True, flip=True, gain=True, color=False):
    """
    Load images for a single class and add to image database.
    :param path: Path to the images with wildcard, e.g. "path/to/*.pgm".
    :param res: Desired resolution as list: [Width, Height].
    :param clss: Current class as integer.
    :param clss_num: Number of all classes.
    :param db: The image database.
    :param rotate: Rotation as augmentation (True/False). Currently unused.
    :param flip: Image flip as augmentation (True/False). Currently unused.
    :param gain: Gain as augmentation (True/False). Currently unused.
    :param color: Color images?
    :return: The updated database.
    """
    files = glob(path)
    for file in files:
        if color:
            img = cv2.imread(file)
        else:
            img = cv2.imread(file, 0)
        try:
            img = cv2.resize(img, (res["x"], res["y"]))
        except:
            print("Error loading image: " + file)
        db.append((img / 255, to_categorical(clss, clss_num), file))
    return db


def finish_db(db, color=False):
    """
    Finish the database after loading. Calculate image mean, shuffle and reshape.
    :param db: The database to finish.
    :param color: Color images?
    :return: Images, ground truth label and image mean.
    """
    random.shuffle(db)
    x, y, p = list(map(np.array, list(zip(*db))))
    mean = np.mean(x)
    x -= mean
    if color:
        x = x.reshape(*x.shape)
    else:
        x = x.reshape(*x.shape, 1)
    return x, y, mean
