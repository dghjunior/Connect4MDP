import tensorflow as tf
from keras import layers
from keras import Model
import numpy as np

class DQN(Model):
    """ A class representing a deep Q network for connect 4. """
    
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = layers.Conv2D(64, 4, activation="relu", input_shape=(6, 7, 1))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.dense3 = layers.Dense(7)

        
    
    def call(self, inputs):
        
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return x
    
    def get_action(model, observation, available_moves=[0,1,2,3,4,5,6]):
        if len(available_moves) == 0:
            return None, None
        else:
            observation = np.array(observation).reshape(1,6,7,1)
            logits = model.predict(observation)
            prob_weights = tf.nn.softmax(logits).numpy()
            action = list(prob_weights[0]).index(max(prob_weights[0]))
                
            return action, prob_weights[0]