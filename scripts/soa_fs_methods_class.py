import numpy as np     
import pandas as pd
import random
import json

# Sklearn modules
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

# XGBoost packages
import xgboost as xgb
from xgboost.sklearn import XGBRegressor  

# User defined files and classes
from read_data import inputs
from predictor_models import predictor_models
import utils_dataset as utilsd

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

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


class soa_methods():
    def __init__(self, XX, YY, test_size=0.1,random_state=40, **kwargs):
        self.test_size = test_size
        self.random_state = random_state 
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(XX, YY, 
                                                                                test_size=self.test_size, 
                                                                                 random_state=self.random_state)
        if 'X_test' in kwargs.keys():
            self.X_test = kwargs['X_test']
            self.y_test = kwargs['y_test']
    
    # XGBoost
    def xgboost(self, X_stand, Y_stand, descriptors, onlyImportant=False, 
                     plot_fig=True, save_fig=False, fig_name='xgboost.pdf'):
        
        clf = XGBRegressor(n_estimators=100, learning_rate=0.025, max_depth=20, verbosity=0, booster='gbtree', 
            reg_alpha=np.exp(-6.788644799030888), reg_lambda=np.exp(-7.450413274554533), 
            gamma=np.exp(-5.374463422208394), subsample=0.5, objective= 'reg:squarederror', n_jobs=1)
        
        paras = clf.get_params()
        clf.fit(self.X_train, self.y_train)   
        
        score = clf.score(self.X_train, self.y_train)
        print("Training score: ", score)

        scores = cross_val_score(clf, self.X_train, self.y_train,cv=2) #cv=10
        print("Mean cross-validation score: %.2f" % scores.mean())

        ypred = clf.predict(self.X_test)
        mse_test = mean_squared_error(self.y_test, ypred)
        print("MSE: %.2f" % mse_test)
        print("RMSE: %.2f" % (mse_test**(1/2.0)))

        f_importance = clf.get_booster().get_score(importance_type='gain')
        feature_importance_dict={}
        
        if onlyImportant:
            for f,value in f_importance.items():
                feature_index = int(f.split('f')[1])
                feature_importance_dict[descriptors[feature_index]] = value
                print(f"Column: {feature_index}, descriptor: {descriptors[feature_index]}")
        
        # XGBoost gives scores only for features that were retained
        # The following peice of code sets the score to 0 for the remaining features
        elif not onlyImportant:  
            num_features = np.linspace(0,len(descriptors)-1,len(descriptors), dtype=int)
            num_features_found = []

            for f,value in f_importance.items():
                feature_index = int(f.split('f')[1])
                num_features_found.append(feature_index)       

            num_features_notFound = np.setdiff1d(num_features,num_features_found).tolist()

            for f in num_features_notFound:
                f_importance['f'+str(f)] = 0.0

            for f in num_features:
                feature_importance_dict[descriptors[f]] = f_importance['f'+str(f)]       
        
        importance_df = pd.DataFrame.from_dict(data=feature_importance_dict, orient='index')
        
        if plot_fig:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.barh(descriptors,importance_df[0])
            ax.set_xlabel('XGBoost Feature Importance')
            ax.set_ylabel('Features')
            ax.set_yticklabels(descriptors)  
            if save_fig:
                plt.tight_layout()
                plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    
        return feature_importance_dict, importance_df, mse_test  
    
    # LASSO
    def lasso(self,X_stand,Y_stand,descriptors,onlyImportant=False,
              plot_fig=True, save_fig=False, fig_name='lasso.pdf'):
        
        X_train, X_test, y_train, y_test = train_test_split(X_stand, Y_stand, test_size=self.test_size, random_state=self.random_state)
        alpha_range=np.arange(0.0001,0.01,0.0001)
        
        pipeline = Pipeline([('scaler',StandardScaler()),('model',Lasso())])
        search = GridSearchCV(pipeline,{'model__alpha':alpha_range},cv = 5, scoring="neg_mean_squared_error")
        search.fit(X_train,y_train)
        lasso_parameters = search.best_params_
        coefficients = search.best_estimator_.named_steps['model'].coef_
        importance = np.abs(coefficients)
        feature_importance_dict = {}
        
        # Generalization Error --> Used to assigning RL rewards
        ypred_test = search.predict(X_test)
        mse_test = mean_squared_error(y_test, ypred_test)
        
        if onlyImportant:
            dict_keys = descriptors[importance > 0.0]
            dict_values = importance[importance > 0.0]
            for i in range(0,len(dict_keys)):
                feature_importance_dict[dict_keys[i]] = dict_values[i]
        elif not onlyImportant:
            for i in range(0,len(descriptors)):
                feature_importance_dict[descriptors[i]] = importance[i]
        
        if plot_fig:
            fig, ax = plt.subplots(figsize=(6, 4.5))
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

if __name__=="__main__":
    
    run_folder = '/Users/maitreyeesharma/WORKSPACE/PostDoc/Chemistry/SPIRAL/codes/RL/RL_FS/'
    
    # Reading the input json file with dataset filename and path information
    with open(run_folder+'scripts/inputs.json', "r") as f:
        input_dict = json.load(f)

    input_type = input_dict['InputType']
    input_path = input_dict['InputPath']
    input_file = input_dict['InputFile']
    output_dir = input_dict['OutputDirectory']
    file_name_prefix=output_dir+input_file.split('.csv')[0] 

    input_data = inputs(input_type=input_type,
                               input_path=input_path,
                               input_file=input_file)

    X_data, Y_data, descriptors = input_data.read_inputs()
    X_stand, X_stand_df, scalerX = utilsd.standardize_data(X_data)
    Y_stand, Y_stand_df, scalerY = utilsd.standardize_data(pd.DataFrame({'target':Y_data[:,0]}))
    
    methods = soa_methods(X_stand,Y_stand)

    feature_importance_dict_xgboost, importance_df_xgboost, mse_test_xgboost = methods.xgboost(X_stand, Y_stand, descriptors,
                                                                                save_fig=True,fig_name=file_name_prefix+'_xgboost.pdf')
    feature_importance_dict_lasso, importance_df_lasso, mse_test_lasso = methods.lasso(X_stand, Y_stand, descriptors,
                                                                                save_fig=True,fig_name=file_name_prefix+'_lasso.pdf')

    print(mse_test_xgboost)
    print(mse_test_lasso)
    mse_df = pd.DataFrame([[mse_test_xgboost,mse_test_lasso]],columns=['MSE_xgboost','MSE_lasso'])
    importance_df_xgboost.to_csv(file_name_prefix+'_xgboost.csv')
    importance_df_lasso.to_csv(file_name_prefix+'_lasso.csv')
    mse_df.to_csv(file_name_prefix+'_soa_mse.csv')
