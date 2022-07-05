from unittest import TestCase
from applications.daimler.gen_imdb_daimler import generate
from applications.daimler.train import train

class Tests(TestCase):
    def test_daimler(self):
        generate("db", {"x": 18, "y": 36}, "img.db")
        train("img.db", "model.h5")

if __name__ == "__main__":
    t = Tests()
    t.test_daimler()