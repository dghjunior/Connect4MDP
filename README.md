# Connect4MDP
A Deep Q Learning network designed to play Connect4 against a user.

### Authors
Kyle Becker & Daniel Harper

## Requirements
- Python 3.x
- Tensorflow 2.x
- Numpy
- Tkinter

## Description
This project consists of the following parts:
- a GUI that allows for PvP games of Connect 4
- a GUI that allows for PvE games of Connect 4 against a Deep Q Network
- A program to train the Deep Q Network against a random opponent
- A program to train 2 Deep Q Networks against each other

The DQN is implemented in Keras, and is a Convolutional Neural Net with a single convolution layer that creates 64 filters of size (4x4) to capture features of the board in order to reduce the size of the state space. It outputs the expected utility of each available move \[0,6\].

The training loop has the network play through a series of Connect 4 games against either a random opponent or a second network. During each game, every (state, action) pair is recorded, and after each game, all (state, action) pairs are assigned a reward value based on the outcome of the game (win, loss, or tie). Loss is calculated between the predicted and assigned reward for a given action, then the gradients are backprogated through the network.

### Model Architecture
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 3, 4, 64)          1088      
                                                                 
 re_lu (ReLU)                (None, 3, 4, 64)          0         
                                                                 
 flatten (Flatten)           (None, 768)               0         
                                                                 
 dense (Dense)               (None, 64)                49216     
                                                                 
 re_lu_1 (ReLU)              (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 64)                4160      
                                                                 
 re_lu_2 (ReLU)              (None, 64)                0         
                                                                 
 dense_2 (Dense)             (None, 7)                 455       
                                                                 
=================================================================
Total params: 54,919
Trainable params: 54,919
Non-trainable params: 0
_________________________________________________________________
```
## Train the Model
This step is not required, but if you want to run a training session for the model, use one of the following commands:

```bash
$ python train.py
$ python trainNvN.py
```
Running `train.py` will train the network against a random opponent, while `trainNvN.py` will train 2 identical networks against each other.

## Play the Game
To play against another player:
```bash
$ python PvP.py
```
To play against a trained network:
```bash
$ python PvE
```

## Acknowledgments
The code for the training loop is borrowed heavily from [this Towards Data Science article](https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca) by Lee Schmalz, titled "Playing Connect 4 with Deep Q-Learning."
