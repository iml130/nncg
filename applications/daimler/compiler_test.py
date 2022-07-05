from unittest import TestCase
from applications.daimler.gen_imdb_daimler import generate

class Tests(TestCase):
    def test(self):
        generate("db", {"x": 18, "y": 36}, "img.db")

if __name__ == "__main__":
    t = Tests()
    t.test()