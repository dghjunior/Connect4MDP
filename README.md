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
