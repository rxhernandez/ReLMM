import numpy as np
import pandas as pd
# Scikit learn packages for model fitting and scores
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# XGBoost packages
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt

class Predictors:
    def __init__(self,test_size=0.1,random_state=40):
        print('This class contains the predictor models to generate rewards for agents')
        self.test_size=test_size
        self.random_state=random_state
        
    def xgboost(self, x_stand, y_stand, selected_descriptors, all_descriptors, only_important=False, plot_fig=False):
        """
        This function provides XGBoost as the predictor.
   
        XGBoost gives scores only for features that were retained which is what
        the output of the function is if only_important=True.
        The second if condition is the code that sets the score to 0 
        for the remaining features.

        Args: 
            x_stand: Dataframe of input data to the predictor
            y_stand: Dataframe of target property which is the output/label for the predictor
            selected_descriptors: List of selected descriptors at the current RL step
            all_descriptors: All descriptors in the dataset
            onlyImportant: Output the feature importance dictionary with only features selected by XGBoost
                            XGBoost gives scores only for features that were retained which is what
                            the output of the function is if onlyImportant=True.
                            The second if condition is the code that sets the score to 0 
                            for the remaining features.
            plot_fig: True if you want to plot the feature importance bar chart
        """
        x_train, x_test, y_train, y_test = train_test_split(x_stand, y_stand,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        
        clf = XGBRegressor(n_estimators=100, learning_rate=0.025, max_depth=20, verbosity=0, booster='gbtree',
            reg_alpha=np.exp(-6.788644799030888), reg_lambda=np.exp(-7.450413274554533),
            gamma=np.exp(-5.374463422208394), subsample=0.5, objective= 'reg:squarederror', n_jobs=1)

        _ = clf.get_params()
        clf.fit(x_train, y_train)   
        _ = clf.score(x_train, y_train)
        _ = cross_val_score(clf, x_train, y_train,cv=2)
        ypred_train = clf.predict(x_train)
        _ = mean_squared_error(y_train, ypred_train)
        # Generalization Error --> Used to assigning RL rewards
        ypred_test = clf.predict(x_test)
        mse_test = mean_squared_error(y_test, ypred_test)
        # Get the importance of each feature
        f_importance = clf.get_booster().get_score(importance_type='gain')
        _ = max(f_importance.values())
        feature_importance_dict={}
        # If user selects only_important then store only the features in f_importance
        if only_important:
            for f, value in f_importance.items():
                feature_importance_dict[f] = value
        else:
            for desc in all_descriptors:
                if desc in f_importance.keys():
                    feature_importance_dict[desc] = f_importance[desc]
                else:
                    feature_importance_dict[desc] = 0.0
        # Store the feature importance in a dataframe
        importance_df = pd.DataFrame.from_dict(data=feature_importance_dict, orient='index')
        # Plot the feature importance if plot_fig is True
        if plot_fig:
            _, ax = plt.subplots(figsize=(6, 4.5))
            ax.barh(all_descriptors,importance_df[0])
            ax.set_xlabel('XGBoost Feature Importance')
            ax.set_ylabel('Features')
            ax.set_yticklabels(all_descriptors)
        return feature_importance_dict, mse_test
    