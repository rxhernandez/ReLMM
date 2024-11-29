import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

from ReLMM.read_data import Inputs
from ReLMM.utils_dataset import standardize_data

class TestStandardizeData(unittest.TestCase):
    def test_standardize_data(self):
        input_data = Inputs(input_type="SynthData",
                            input_path="../synthetic_dataset",
                            input_file="synthetic_data_randomSamples_200_nonlinearf5.csv")
        x_data, y_data, descriptors = input_data.read_inputs()
        x_stand_DT, x_stand_DT_df, scalerX = standardize_data(x_data)
        scaled_x = np.array([1.38491146,
                            0.14454519,
                            -1.47756123,
                            1.11814321,
                            0.42182201,
                            1.52784061,
                            -0.21586941,
                            1.34187484,
                            1.66013288,
                            0.94720978,
                            -0.32855355,
                            1.54881923])
        self.assertTrue(np.array_equal(np.round(x_stand_DT[0,:], decimals=8), scaled_x))

    def test_initial_training_data(self):
        pass

if __name__ == '__main__':
    unittest.main()