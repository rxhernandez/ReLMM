import unittest

import torch
import torch.nn as nn

class TestQNetwork(unittest.TestCase):
    def test_forward(self):
        state_size = 10
        action_size = 2
        model = nn.Sequential(nn.Linear(state_size, 32),
                              nn.ReLU(),
                              nn.Linear(32, action_size))
        self.assertEqual(model(torch.ones(10)).shape, torch.Size([2]))

if __name__ == '__main__':
    unittest.main()