This file gives instructions on how to run the ReLMM code.

Prerequisites
----------------
The python packages required to run the code can be found in the ReLMM_env.yml file. This file can be run by typing:

* conda env create -f ReLMM_env.yml
* conda activate ReLMM_env

Running
----------------
The code can be run using the single-player_multi-agent_RL.ipynb notebook file.

* Inputs: These can be set in the inputs.json file.
* The second cell in single-player_multi-agent_RL.ipynb contains hyperparameters to set up the ReLMM run. These include number of steps (N_steps) per episode, total number of episodes (total_episodes), training parameters (epsilon, epsilon_decay, gamma and learning rate)
* Output directory: The output directory is automatically created in the ./output/ folder in the code. 