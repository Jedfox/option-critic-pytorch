### Soft Option Critic
Seen below is the readme for the option critic pytorch repository which we forked for the Midterm/Final Implementation. Our changes includes soft_option_critic.py, which includes the architecture implementation, main.py, which trains the models, and Notebook.ipynb, which generates a tensorboard and animations on each environment. 

## CartPole
This is a simple control environment to test speed of conversion. To run it enter command below:

```
python main.py --model option_critic
```
To use the SOC architecture change flag to soft_option_critic

## Four Room experiment
Four-Rooms tests Transfer learning capabilities.  To run it enter command below:

```
python main.py --model option_critic --switch-goal True --env fourrooms
```
To use the SOC architecture change flag to soft_option_critic

## LunarLander experiment
This is a more complex continous control environment. It tests option diversification and training under limited step count. To run it enter command below:

```
python main.py --model option_critic --env LunarLander-v2 --num-options 4 --max_steps_total 500000
```
To use the SOC architecture change flag to soft_option_critic

# Option Critic
This repository is a PyTorch implementation of the paper "The Option-Critic Architecture" by Pierre-Luc Bacon, Jean Harb and Doina Precup [arXiv](https://arxiv.org/abs/1609.05140). It is mostly a rewriting of the original Theano code found [here](https://github.com/jeanharb/option_critic) into PyTorch, focused on readability and ease of understanding the logic begind the Option-Critic.


## CartPole
Currently, the dense architecture can learn CartPole-v0 with a learning rate of 0.005, this has however only been tested with two options. (I dont see any reason to use more than two in the cart pole environment.) Run it as follows:

```
python main.py
```

## Atari Environments
I suspect it will take a grid search over learning rate to work on Pong and such. Just supply the right `--env` argument and the model should switch to convolutions if the environment is Atari compatible.

## Four Room experiment
There are plenty of resources to find a numpy version of the four rooms experiment, this one is a little bit different; represent the state as a one-hot encoded vector, and learn to solve this grid world using a deep net. Run this experiment as follows

```
python main.py --switch-goal True --env fourrooms
```

## Requirements

```
pytorch>=1.12.1
tensorboard>=2.0.2
gym[atari,accept-rom-license,box2d]>=0.15.3
numpy<2.0
matplotlib>=3.5.0
ipython>=8.0.0
jupyter>=1.0.0
```
