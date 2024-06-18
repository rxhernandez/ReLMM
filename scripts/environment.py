import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Define the cooperative environment with two agents
class Environment:
    def __init__(self,state_size,action_size,N_agents,N_steps):
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Number of possible actions per agent
        self.N_agents = N_agents  # Number of agents
        self.N_steps = N_steps # Number of steps to take per episode        
        self.reset()

    def reset(self):
        # Reset environment to initial state
        self.state = np.ones(self.state_size)
        self.steps = 0
        return self.state


    def step(self, actions):
        '''
        This function returns the reward,
        the new state of the environment,
        and number of steps taken for the current
        episode based on the action taken.
        Input: 
            action vector for all agents
        Output: 
            new state
            reward
            done
        '''
        self.state = self.update_state(actions)
        self.steps += 1
        done = self.steps >= self.N_steps  # End after N steps
        return self.state, done

    def update_state(self, actions):
        '''
        This function returns the new state of the environment
        based on the action taken.
        Input: 
            action vector
        Output: 
            new state
        '''
        new_state = actions  # Update state based on actions
        return new_state
    
    def get_rewards(self,predictor_model,X_stand,Y_stand):
        '''
        This function computed the reward
        based on the action taken.
        Input:
            Data
        Output: 
            reward
        '''
        descriptors = X_stand.columns
        sampled_descriptors = [i for i, x in enumerate(self.state) if x == 1]
        X_stand_predictor = pd.DataFrame(X_stand, columns=descriptors[sampled_descriptors])
        Y_stand_predictor = Y_stand

        # Predictor rewards data for training policy
        feature_importance_dict, mse_game = predictor_model.xgboost(X_stand_predictor,Y_stand_predictor,
                                                    sampled_descriptors,descriptors,onlyImportant=False,plot_fig=False)
        reward = 1.0/mse_game  
        all_rewards = np.repeat(reward,self.N_agents)

        return all_rewards 
    
    def get_rewards_test(self,predictor_model,X_stand,Y_stand):
        '''
        This function computed the reward
        based on the action taken.
        Input:
            Data
        Output: 
            reward
        '''
        descriptors = X_stand.columns
        sampled_descriptors = [i for i, x in enumerate(self.state) if x == 1]
        X_stand_predictor = pd.DataFrame(X_stand, columns=descriptors[sampled_descriptors])
        Y_stand_predictor = Y_stand

        # Predictor rewards data for training policy
        feature_importance_dict, mse_game = predictor_model.xgboost(X_stand_predictor,Y_stand_predictor,
                                                              sampled_descriptors,descriptors,onlyImportant=False,plot_fig=False)
        reward = 1.0/mse_game  
        all_rewards = np.repeat(reward,self.N_agents)

        return all_rewards, feature_importance_dict, mse_game 
    