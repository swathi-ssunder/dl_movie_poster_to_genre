import unittest
from utils.data_util import load_data

class TestDataUtil(unittest.TestCase):

    def test_load_data(self):

        data, label, genre_decoder = load_data(limit=100)
        self.assertEquals(data.shape, (100, 3, 268, 182))
        self.assertEquals(len(genre_decoder), 29)
        data, label, _ = load_data(limit=100, size=128)
        self.assertEquals(data.shape, (100, 3, 128, 128))


if __name__ == '__main__':
    unittest.main()
