import unittest

import pandas as pd

from ReLMM.read_data import Inputs
from ReLMM.utils_dataset import standardize_data
from ReLMM.soa_feature_selectors import SOAFeatureSelectors

class TestSOAFeatureSelectors(unittest.TestCase):
    def test_xgboost(self):
        input_data = Inputs(input_type="SynthData",
                            input_path="../synthetic_dataset",
                            input_file="synthetic_data_randomSamples_200_nonlinearf5.csv")
        x_data, y_data, descriptors = input_data.read_inputs()
        x_stand, _, _ = standardize_data(x_data)
        y_stand, _, _ = standardize_data(pd.DataFrame({'target':y_data[:,0]}))
        soa_feature_selectors = SOAFeatureSelectors(x_stand,
                                                    y_stand,
                                                    test_size=0.1,
                                                    random_state=40)
        calc_feat_imp_dict, _, calc_mse_game = soa_feature_selectors.xgboost(descriptors,
                                                                             only_important=False,
                                                                             plot_fig=False,
                                                                             save_fig=False,
                                                                             fig_name='xgboost.pdf')
        
        true_feat_imp_dict = {'z1': 0.17879385,
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
        self.assertTrue({k:round(v, 8) for k, v in calc_feat_imp_dict.items()} == true_feat_imp_dict)
        self.assertTrue(round(calc_mse_game, 8) == true_mse_game)

    def test_lasso(self):
        input_data = Inputs(input_type="SynthData",
                            input_path="../synthetic_dataset",
                            input_file="synthetic_data_randomSamples_200_nonlinearf5.csv")
        x_data, y_data, descriptors = input_data.read_inputs()
        x_stand, _, _ = standardize_data(x_data)
        y_stand, _, _ = standardize_data(pd.DataFrame({'target':y_data[:,0]}))
        soa_feature_selectors = SOAFeatureSelectors(x_stand,
                                                    y_stand,
                                                    test_size=0.1,
                                                    random_state=40)
        calc_feat_imp_dict, _, calc_mse_game = soa_feature_selectors.lasso(descriptors,
                                                                        only_important=False,
                                                                        plot_fig=False,
                                                                        save_fig=False,
                                                                        fig_name='lasso.pdf')


if __name__ == "__main__":
    unittest.main()