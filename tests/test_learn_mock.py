import unittest
from unittest import mock

class Count(object):
    def add(self, a, b):
        return a+b

def add_and_multiply(x, y):
    addition = x + y
    multiple = multiply(x, y)
    return (addition, multiple)

def multiply(x, y):
    return x * y + 3

class TestCount(unittest.TestCase):
    def test_add(self):
        count = Count()
        count.add = mock.Mock(return_value=13, side_effect=count.add)
        result = count.add(8, 8)
        count.add.assert_called_with(8,8)
        self.assertEqual(result, 16)


class MyTestCase(unittest.TestCase):

    @mock.patch("test_learn_mock.multiply")
    def test_add_and_multiply(self, m):
        m.return_value = 15
        x = 3
        y = 5
        addition, multiple = add_and_multiply(x, y)
        m.assert_called_once_with(3, 5)
        self.assertEqual(8, addition)
        self.assertEqual(15, multiple)

if __name__ == "__main__":
    unittest.main()