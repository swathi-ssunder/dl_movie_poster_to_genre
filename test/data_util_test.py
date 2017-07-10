import unittest
from utils.data_util import load_data

class TestDataUtil(unittest.TestCase):

    def test_load_data(self):

        data, label = load_data(limit=100)
        self.assertEquals(data.shape, (100, 3, 268, 182))
        data, label = load_data(limit=100, resize=True)
        self.assertEquals(data.shape, (100, 3, 182, 182))


if __name__ == '__main__':
    unittest.main()
