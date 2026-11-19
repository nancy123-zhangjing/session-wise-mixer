import unittest
from scripts.kuairec_data.l5o import user_slates

class UserSlatesTest(unittest.TestCase):
    def test_contiguous_distinct(self):
        items = list(range(1, 21))  # 20 distinct events
        s = user_slates(items, 5, 5)
        self.assertEqual(s[0], ([15, 16, 17, 18, 19], 15))  # test = last 5
        self.assertEqual(s[1], ([10, 11, 12, 13, 14], 10))  # validation
        self.assertEqual([x[1] for x in s], [15, 10, 5])     # last train needs history >= 5

    def test_repeat_inside_window_preserved(self):
        items = [1, 2, 3, 4, 5, 6, 7, 9, 8, 9]  # 9 repeated inside final window
        kept, start = user_slates(items, 5, 2)[0]
        self.assertEqual(kept, [5, 6, 7, 8, 9])
        self.assertEqual([items[i] for i in kept], [6, 7, 9, 8, 9])
        self.assertEqual(start, 5)

    def test_repetition_never_moves_boundary(self):
        items = [10, 11, 12, 1, 2, 1, 3, 4]
        kept, start = user_slates(items, 5, 1)[0]
        self.assertEqual([items[i] for i in kept], [1, 2, 1, 3, 4])
        self.assertEqual(kept, [3, 4, 5, 6, 7])
        self.assertEqual(start, 3)

    def test_minimum_user_has_no_supervised_train_block(self):
        self.assertEqual(len(user_slates(list(range(15)), 5, 5)), 2)
        self.assertEqual(len(user_slates(list(range(20)), 5, 5)), 3)

    def test_stops_when_history_short(self):
        self.assertEqual(user_slates([1, 2, 3, 4, 5, 6, 7], 5, 5), [])


if __name__ == '__main__':
    unittest.main()
