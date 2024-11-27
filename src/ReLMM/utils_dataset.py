import random

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import pandas as pd

# Add slicing of the input XX tensor with additional input for the columns picked out by XGBoost or other feature selection methods
class InputDataset(Dataset):
    """ Input dataset used for training """

    def __init__(self, XX, YY, descriptors=[None], transform=None):
        """
        Args:
            XX: NN Input features vector
            YY: NN Labels
            descriptors(list of strings): Names of the input features
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.XX = torch.tensor(XX).to(torch.float32) # Transforming input matrix to tensor
        # self.YY = YY
        if torch.is_tensor(YY):
            self.YY = YY 
        else:
            self.YY = torch.tensor(YY).to(torch.float32) # Transforming input matrix to tensor
        self.descriptors = descriptors
        self.transform = transform

    def __len__(self):
        return self.XX.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.XX[idx,:]
        y = self.YY[idx] #self.YY[idx,:]
        item = {'in_features':x,'labels':y}
        return item

def standardize_data(x):
    """
    Args:
        XX: NN Input features vector as a dataframe
    """   
    scalerX = StandardScaler().fit(x)
    x_train = scalerX.transform(x)
    x_train_df = pd.DataFrame(x_train,columns=x.columns)
    return x_train, x_train_df,scalerX


def initial_training_data(expert_trainer,X_stand,Y_stand,descriptors,num_game_instances = 50):
    state = []
    reward = []
    for iGame in range(0,num_game_instances):
        sampled_descriptors = random.sample(list(descriptors), 5)
        X_stand_training = pd.DataFrame(X_stand, columns=sampled_descriptors)
        Y_stand_training = Y_stand

        # XGboost data for training policy
        feature_importance_dict, reward_game = expert_trainer.expert_xgboost(X_stand_training,Y_stand_training,sampled_descriptors,onlyTopChoices=False)

        state_game = []
        for idescriptor in range(0,len(list(descriptors))):
            if descriptors[idescriptor] in feature_importance_dict.keys():
                state_game.append(int(1))
            else:
                state_game.append(int(0))
        
        state.append(state_game)
        reward.append(reward_game)
        
    return state, reward