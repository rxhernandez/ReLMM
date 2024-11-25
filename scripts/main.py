import numpy as np
import pandas as pd
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# User defined files and classes
import sys
from read_data import inputs
import utils_dataset as utilsd
from environment import Environment
from qlearning import QNetwork
from predictor_models import predictor_models

## Main Function

# Reading the input json file with dataset filename and path information
with open('inputs.json', "r") as f:
    input_dict = json.load(f)

run_folder = input_dict['RunFolder']
input_type = input_dict['InputType']
input_path = input_dict['InputPath']
input_file = input_dict['InputFile']
output_dir = input_dict['OutputDirectory']

# Create a new output directory if it does not exist
isExist = os.path.exists(output_dir)
if not isExist:
    os.makedirs(output_dir)
    print("The new directory is created!", output_dir)

input_data = inputs(input_type=input_type,
                           input_path=input_path,
                           input_file=input_file)

X_data, Y_data, descriptors = input_data.read_inputs()
X_stand_all, X_stand_df_all, scalerX = utilsd.standardize_data(X_data)
Y_stand_all, Y_stand_df_all, scalerY = utilsd.standardize_data(pd.DataFrame({'target':Y_data[:,0]}))
X_stand_df, X_test_df, Y_stand_df, Y_test_df = train_test_split(X_stand_df_all, Y_stand_df_all, test_size=0.1, random_state=0)
X_stand, X_test, Y_stand, Y_test = train_test_split(X_stand_all, Y_stand_all, test_size=0.1, random_state=0)

# Dataset parameters
total_num_features = len(descriptors)

# Environment parameters
state_size = total_num_features  # Size of the state space
N_agents = total_num_features # Number of agents
action_size = 2  # Number of possible actions
N_steps = 100 # Number of steps to take per episode
predictor_model = predictor_models()

# Hyperparameters
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Decay rate of exploration
gamma = 0.95  # Discount factor
learning_rate = 0.001

# Initialize environment and Q-networks for each agent
env = Environment(state_size,action_size,N_agents,N_steps)
agent_model = {}
agent_optimizer = {}
agent_qvalue = {}

for i_agent in range(N_agents):
    model_name = 'agent'+str(i_agent)+'_model'
    optimizer_name = 'agent'+str(i_agent)+'_optimizer'
    agent_model[model_name] = QNetwork(env.state_size, env.action_size)
    agent_optimizer[optimizer_name] = optim.Adam(agent_model[model_name].parameters(), lr=learning_rate)

# Training loop
total_episodes = 1000

for episode in range(total_episodes):
    state = env.reset()
    total_rewards = np.zeros(N_agents) # Total rewards for agents
    
    while True:
        # Agents choose actions using epsilon-greedy policy
        if np.random.rand() <= epsilon:
            actions = np.random.randint(2, size=(N_agents,))  # Random actions
        else:
            with torch.no_grad():
                actions_list = []
                for i_agent in range(N_agents):
                    model_name = 'agent'+str(i_agent)+'_model'
                    q_values = agent_model[model_name](torch.tensor(state, dtype=torch.float32))
                    actions_list.append(torch.argmax(q_values).item())
                actions = np.array(actions_list)
                
        if all(action == 0 for action in actions):
            non_zero_action = np.random.randint(N_agents)
            actions[non_zero_action] = 1
        
        # Take actions and observe next states, rewards, done
        next_state, done = env.step(actions)
        rewards = env.get_rewards(predictor_model,X_stand_df,Y_stand_df)

        # Update Q-values for each agent
        for i, (model, optimizer, reward) in enumerate(zip(agent_model.values(),
                                                           agent_optimizer.values(),
                                                           rewards)):
            q_values_next = model(torch.tensor(next_state, dtype=torch.float32))
            target = reward + gamma * torch.max(q_values_next)

            q_values = model(torch.tensor(state, dtype=torch.float32))
            loss = nn.functional.mse_loss(q_values[actions[i]], target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_rewards[i] += reward
        state = next_state

        if done:
            break

    # Decay epsilon
    if epsilon > 0.01:
        epsilon *= epsilon_decay

    # Print episode results
    if (episode + 1) % 1 == 0:
        print(f"Episode: {episode + 1}, state: {state}, Total Rewards: {total_rewards[0]}, Epsilon: {epsilon}")

        
# Save QNetwork Models
for i_agent in range(N_agents):
    model_name = 'agent'+str(i_agent)+'_model'
    saveModel_filename = output_dir+model_name+'.pt'
    torch.save(agent_model[model_name].state_dict(), saveModel_filename)
    
    
# Testing the trained agents
state = env.reset()
total_rewards = np.zeros(N_agents)

while True:
    with torch.no_grad():
        actions_list = []
        for i_agent in range(N_agents):
            model_name = 'agent'+str(i_agent)+'_model'
            q_values = agent_model[model_name](torch.tensor(state, dtype=torch.float32))
            actions_list.append(torch.argmax(q_values).item())        
        actions = np.array(actions_list)

    next_state, done = env.step(actions)
    rewards, feature_importance_dict_rl, mse_rl = env.get_rewards_test(predictor_model,X_stand_df,Y_stand_df)    
    total_rewards += rewards  
    state = next_state

    if done:
        break

importance_df_rl = pd.DataFrame.from_dict(data=feature_importance_dict_rl, orient='index')
importance_df_rl.to_csv(output_dir+'rl.csv')
print(f"Test Total Rewards: {total_rewards}, state: {state}")
