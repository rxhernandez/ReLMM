import unittest

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from ReLMM.environment import Environment
from ReLMM.predictors import Predictors
from ReLMM.utils_dataset import standardize_data
from ReLMM.read_data import Inputs

class TestEnvironment(unittest.TestCase):
		
	def test_reset(self):
		environment = Environment(10, 2, 10, 100)
		self.assertTrue(np.array_equal(environment.reset(), np.ones(10)))
		self.assertEqual(environment.steps, 0)
	
	def test_update_state(self):
		environment = Environment(10, 2, 10, 100)
		actions = np.zeros(10)
		self.assertTrue(np.array_equal(environment.update_state(actions), actions))

	def test_step(self):
		environment = Environment(10, 2, 10, 100)
		actions = np.zeros(10)
		self.assertTrue(np.array_equal(environment.step(actions)[0], actions))
		self.assertFalse(environment.step(actions)[1])

	def test_get_rewards(self):
		input_data = Inputs(input_type="SynthData",
		 		    input_path="../synthetic_dataset",
				    input_file="synthetic_data_randomSamples_200_nonlinearf5.csv")
		X_data, Y_data, _ = input_data.read_inputs()
		_, X_stand_df_all, _ = standardize_data(X_data)
		_, Y_stand_df_all, _ = standardize_data(pd.DataFrame({'target':Y_data[:,0]}))
		X_stand_df, _, Y_stand_df, _ = train_test_split(X_stand_df_all, Y_stand_df_all, test_size=0.1, random_state=40)
		predictor_model = Predictors(random_state=40)
		environment = Environment(10, 2, 10, 100)
		self.assertTrue(np.array_equal(np.round(environment.get_rewards(predictor_model, X_stand_df, Y_stand_df), 
										  decimals=8),
										np.repeat(0.93575101, 10)))
			
if __name__ == '__main__':
	unittest.main()		
