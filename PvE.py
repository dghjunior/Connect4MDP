import numpy as np
import tensorflow as tf
import models.DQN as DQN

model = DQN()
model.load_weights("runs/50000 epochs/DQN_weights.h5")