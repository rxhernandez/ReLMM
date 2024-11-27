import json

import numpy as np
import pandas as pd
# Sklearn modules
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
# XGBoost packages
from xgboost.sklearn import XGBRegressor
# Plotting libraries
import matplotlib.pyplot as plt

# User defined files and classes
from .read_data import Inputs
from .utils_dataset import standardize_data

# Tick parameters
plt.rcParams.update({
"text.usetex":True,
"font.family":"serif",
"font.serif":["Computer Modern Roman"]})
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.width'] = 1


class SOAFeatureSelectors():
    def __init__(self, x, y, test_size=0.1,random_state=40):
        self.test_size = test_size
        self.random_state = random_state
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)
    
    # XGBoost
    def xgboost(self, descriptors, only_important=False, plot_fig=True, save_fig=False, fig_name='xgboost.pdf'):
        
        clf = XGBRegressor(n_estimators=100, learning_rate=0.025, max_depth=20, verbosity=0, booster='gbtree',
            reg_alpha=np.exp(-6.788644799030888), reg_lambda=np.exp(-7.450413274554533),
            gamma=np.exp(-5.374463422208394), subsample=0.5, objective= 'reg:squarederror', n_jobs=1)
        
        _ = clf.get_params()
        clf.fit(self.x_train, self.y_train) 
        score = clf.score(self.x_train, self.y_train)
        print("Training score: ", score)
        scores = cross_val_score(clf, self.x_train, self.y_train,cv=2) #cv=10
        # print("Mean cross-validation score: %.2f" % scores.mean())
        print(f'Mean cross-validation score: {round(scores.mean(),2)}')

        ypred = clf.predict(self.x_test)
        mse_test = mean_squared_error(self.y_test, ypred)
        print(f"MSE: {round(mse_test,2)}")
        print(f"RMSE: {round(mse_test**(1/2.0),2)}" )

        f_importance = clf.get_booster().get_score(importance_type='gain')
        feature_importance_dict={}
        
        if only_important:
            for feature, value in f_importance.items():
                feature_index = int(feature.split('f')[1])
                feature_importance_dict[descriptors[feature_index]] = value
                print(f"Column: {feature_index}, descriptor: {descriptors[feature_index]}")
        # XGBoost gives scores only for features that were retained
        # The following peice of code sets the score to 0 for the remaining features
        else:
            num_features = np.linspace(0,len(descriptors)-1,len(descriptors), dtype=int)
            num_features_found = []
            for feature, value in f_importance.items():
                feature_index = int(feature.split('f')[1])
                num_features_found.append(feature_index)
            num_features_not_found = np.setdiff1d(num_features,num_features_found).tolist()
            for feature in num_features_not_found:
                f_importance['f'+str(feature)] = 0.0
            for feature in num_features:
                feature_importance_dict[descriptors[feature]] = f_importance['f'+str(feature)]

        importance_df = pd.DataFrame.from_dict(data=feature_importance_dict, orient='index')
        
        if plot_fig:
            _, ax = plt.subplots(figsize=(6, 4.5))
            ax.barh(descriptors,importance_df[0])
            ax.set_xlabel('XGBoost Feature Importance')
            ax.set_ylabel('Features')
            ax.set_yticklabels(descriptors)  
            if save_fig:
                plt.tight_layout()
                plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        return feature_importance_dict, importance_df, mse_test
    
    # LASSO
    def lasso(self, descriptors, only_important=False, plot_fig=True, save_fig=False, fig_name='lasso.pdf'):
        alpha_range=np.arange(0.0001,0.01,0.0001)
        pipeline = Pipeline([('scaler',StandardScaler()),('model',Lasso())])
        search = GridSearchCV(pipeline,{'model__alpha':alpha_range},cv = 5, scoring="neg_mean_squared_error")
        search.fit(self.x_train, self.y_train)
        _ = search.best_params_
        coefficients = search.best_estimator_.named_steps['model'].coef_
        importance = np.abs(coefficients)
        feature_importance_dict = {}
        
        # Generalization Error --> Used to assigning RL rewards
        ypred_test = search.predict(self.x_test)
        mse_test = mean_squared_error(self.y_test, ypred_test)
        
        if only_important:
            dict_keys = descriptors[importance > 0.0]
            dict_values = importance[importance > 0.0]
            for i in range(len(dict_keys)):
                feature_importance_dict[dict_keys[i]] = dict_values[i]
        else:
            for i in range(len(descriptors)):
                feature_importance_dict[descriptors[i]] = importance[i]
        
        if plot_fig:
            _, ax = plt.subplots(figsize=(6, 4.5))
            importance_df = pd.DataFrame.from_dict(data=feature_importance_dict, orient='index')
            ax.barh(descriptors,importance_df[0])
            ax.set_xlabel('LASSO Coefficients')
            ax.set_ylabel('Features')
            ax.set_yticklabels(descriptors)
            plt.tight_layout()
            if save_fig:
                plt.tight_layout()
                plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        return feature_importance_dict, importance_df, mse_test