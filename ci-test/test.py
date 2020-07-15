import unittest
from test_case import just_test


class Test_Case(unittest.TestCase):
    def test_case(self):
        self.assertEqual(just_test(), 'tmp')
