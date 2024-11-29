import unittest

import numpy as np
import pandas as pd

from ReLMM.read_data import Inputs

class TestInputs(unittest.TestCase):
    def test_read_inputs(self):
        input_data = Inputs(input_type="SynthData",
                    input_path="../synthetic_dataset",
                    input_file="synthetic_data_randomSamples_200_nonlinearf5.csv")
        X_data, Y_data, _ = input_data.read_inputs()
        self.assertEqual(X_data.shape, (200, 18))
        self.assertEqual(Y_data.shape, (200, 1))
        self.assertEqual(np.round(X_data[0,0], 8), 0.0)
        self.assertEqual(np.round(Y_data[0,0], 8), 0.0)

    def test_read_SynthData(self):
        input_data = Inputs(input_type="SynthData",
                    input_path="../synthetic_dataset",
                    input_file="synthetic_data_randomSamples_200_nonlinearf5.csv")
        X_data, Y_data, _ = input_data.read_SynthData()
        self.assertEqual(X_data.shape, (200, 18))
        self.assertEqual(Y_data.shape, (200, 1))
        self.assertEqual(np.round(X_data[0,0], 8), 0.0)
        self.assertEqual(np.round(Y_data[0,0], 8), 0.0)