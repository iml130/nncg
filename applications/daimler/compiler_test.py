from unittest import TestCase
from applications.daimler.gen_imdb_daimler import generate
from applications.daimler.train import train
from applications.daimler.compile import compile

class Tests(TestCase):
    def test_daimler(self):
        generate("db", {"x": 18, "y": 36}, "img.db")
        train("img.db", "model.h5")
        compile("model.h5", ".", "img.db")
    
    def test_daimler_separable(self):
        generate("db", {"x": 18, "y": 36}, "img.db")
        train("img.db", "model_separable.h5", use_separable=True)
        compile("model_separable.h5", ".", "img.db")

if __name__ == "__main__":
    t = Tests()
    t.test_daimler()
