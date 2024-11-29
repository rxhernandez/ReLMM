import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ReLMM.predictors import Predictors
from ReLMM.read_data import Inputs
from ReLMM.utils_dataset import standardize_data   

class TestPredictors(unittest.TestCase):
    
    def test_xgboost(self):
        input_data = Inputs(input_type="SynthData",
                    input_path="../synthetic_dataset",
                    input_file="synthetic_data_randomSamples_200_nonlinearf5.csv")
        X_data, Y_data, _ = input_data.read_inputs()
        _, X_stand_df_all, _ = standardize_data(X_data)
        _, Y_stand_df_all, _ = standardize_data(pd.DataFrame({'target':Y_data[:,0]}))
        predictors = Predictors(random_state=40)
        X_stand_df, _, Y_stand_df, _ = train_test_split(X_stand_df_all, Y_stand_df_all, test_size=0.1, random_state=40)
        descriptors = X_stand_df.columns
        sampled_descriptors = descriptors
        calc_feat_imp, calc_mse_game = predictors.xgboost(X_stand_df,Y_stand_df,
                                        sampled_descriptors,
                                        descriptors,
                                        only_important=False,plot_fig=False)
        true_feat_imp={'z1': 0.17879385,
                       'z2': 0.30998832, 
                       'z3': 0.45630050, 
                       'z4': 0.55083472, 
                       'z5': 0.63767970, 
                       'z6': 0.86131877, 
                       'z7': 0.74008137, 
                       'z8': 0.72119111, 
                       'f1': 0.85768896, 
                       'f2': 1.45315242, 
                       'f3': 2.03433371, 
                       'f4': 1.01865470}
        true_mse_game = 1.29856149
        self.assertTrue({k:round(v, 8) for k, v in calc_feat_imp.items()} == true_feat_imp)
        self.assertTrue(round(calc_mse_game, 8) == true_mse_game)

if __name__ == "__main__":
    unittest.main()