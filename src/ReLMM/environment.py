import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class Environment:
    """
    Define the cooperative environment with two agents
    """
    def __init__(self, state_size, action_size, n_agents, n_steps):
        self.state = None
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Number of possible actions per agent
        self.n_agents = n_agents  # Number of agents
        self.n_steps = n_steps # Number of steps to take per episode        
        self.reset()

    def reset(self):
        """
        Reset environment to initial state
        """
        self.state = np.ones(self.state_size)
        self.steps = 0
        return self.state

    def step(self, actions):
        """
        This function returns the reward,
        the new state of the environment,
        and number of steps taken for the current
        episode based on the action taken.
        Args:
            actions: action vector for all agents 
        Returns: 
            new_state, reward, done
        """
        self.state = self.update_state(actions)
        self.steps += 1
        done = self.steps >= self.n_steps  # End after N steps
        return self.state, done

    def update_state(self, actions):
        """
        This function returns the new state of the environment
        based on the action taken.
        Args: 
            actions : action vector for all agents
        Returns: 
            new_state
        """
        new_state = actions  # Update state based on actions
        return new_state

    def get_rewards(self, predictor_model, x_stand, y_stand):
        '''
        This function computed the reward
        based on the action taken.
        Args:
            predictor_model: Model to predict rewards
            x_stand: Standardized input data
            y_stand: Standardized output data
        Returns: 
            all_rewards
        '''
        descriptors = x_stand.columns
        sampled_descriptors = [i for i, x in enumerate(self.state) if x == 1]
        x_stand_predictor = pd.DataFrame(x_stand, columns=descriptors[sampled_descriptors])
        y_stand_predictor = y_stand
        # Predictor rewards data for training policy
        _, mse_game = predictor_model.xgboost(x_stand_predictor,y_stand_predictor,
                                              sampled_descriptors,descriptors,only_important=False,plot_fig=False)
        reward = 1.0/mse_game
        all_rewards = np.repeat(reward,self.n_agents)
        return all_rewards
    
    def get_rewards_test(self, predictor_model, x_stand, y_stand):
        """
        This function computed the reward
        based on the action taken.
        Args:
            predictor_model: Model to predict rewards
            x_stand: Standardized input data
            y_stand: Standardized output data
        Returns: 
            reward
        """
        descriptors = x_stand.columns
        sampled_descriptors = [i for i, x in enumerate(self.state) if x == 1]
        X_stand_predictor = pd.DataFrame(x_stand, columns=descriptors[sampled_descriptors])
        Y_stand_predictor = y_stand

        # Predictor rewards data for training policy
        feature_importance_dict, mse_game = predictor_model.xgboost(X_stand_predictor,Y_stand_predictor,
                                                              sampled_descriptors,descriptors,only_important=False,plot_fig=False)
        reward = 1.0/mse_game  
        all_rewards = np.repeat(reward,self.n_agents)

        return all_rewards, feature_importance_dict, mse_game 
    