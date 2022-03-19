import unittest


class TestDummy(unittest.TestCase):

    def test_dummy_1(self):
        a = 2 + 2
        self.assertEquals(a, 4)

if __name__ == "__main__":
    unittest.main()