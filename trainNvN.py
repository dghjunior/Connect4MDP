import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from Connect4 import Connect4
from Memory import Memory
from models.DQN import DQN

"""
trainNvN.py

Training loop for training 2 DQN networks against each other. Code is borrowed heavily from
https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca,
with modifications to fit our game representation and network structure.
"""

# Pulled from TDS article
def compute_loss(logits, actions, rewards): 
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss

#@tf.function
def train_step(model, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
      # Forward propagate through the agent network
        
        logits = model(observations)
        loss = compute_loss(logits, actions, rewards)
        grads = tape.gradient(loss, model.trainable_variables)
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
def get_action(model, observation, epsilon, available_moves=[0,1,2,3,4,5,6]):
    if len(available_moves) == 0:
        return None, None
    else:
        #determine whether model action or random action based on epsilon
        act = np.random.choice(['model','random'], 1, p=[1-epsilon, epsilon])[0]
        observation = np.array(observation).reshape(1,6,7,1)
        logits = model.predict(observation)
        prob_weights = tf.nn.softmax(logits).numpy()
        
        if act == 'model':
            action = list(prob_weights[0]).index(max(prob_weights[0]))
        if act == 'random':
            action = np.random.choice(available_moves)
            
        return action, prob_weights[0]


def random_turn(connect4):
    if len(connect4.available_moves()) == 0:
        return
    else:
        col = np.random.choice(connect4.available_moves())
        connect4.drop_piece(col)

def encode_board(board):
    new_board = np.empty([6,7,1], dtype=np.float64())
    encoding = {'e': 0, 'r': 1, 'y': 2}
    for row in range(6):
        for col in range(7):
            new_board[row][col] = encoding[board[row][col]]
    return new_board

### Training of model

env = Connect4()

### Instatiate models
model_p1 = DQN()
model_p2 = DQN()
memory = Memory()
memory2 = Memory()
num_episodes = 50000
epsilon = 1

reward = 0
win_count = 0

optimizer1 = keras.optimizers.Adam(0.001)
optimizer2 = keras.optimizers.Adam(0.001)

train_log = {}

for episode in range(num_episodes):
    
    # Start new game
    env.new_board()
    observation = encode_board(env.board)
    memory.clear()
    
    epsilon = epsilon * .9998
    
    # Play game
    while True:
        
        # Let both networks pick a move
        action, _ = get_action(model_p1, observation, epsilon, 
                               available_moves=env.available_moves())
        
        env.drop_piece(action)
        
        action2, _ = get_action(model_p2, observation, epsilon,
                               available_moves=env.available_moves())
        
        env.drop_piece(action2)
        
        observation = encode_board(env.board)
        
        # Check if game is over
        if env.check_tie():
            reward = 1
            done = [True, 'tie']
        else:
            done = env.check_win()
        
        # Set rewards for wins and losses
        if not done[0]:
            reward = 0
        elif 'y' in done[1]: # random player wins
            reward = -20
        elif 'r' in done[1]: # network wins
            win_count += 1
            reward = 20
            
        # TODO: deal with ties ?

        memory.add_to_memory(observation, action, reward)
        reward = np.int32(-reward)
        memory2.add_to_memory(observation, action2, reward)
        
        
        if done[0]:
            
            train_log[episode] = (win_count, win_count / (episode+1))
            
            # train network
            train_step(model_p1, optimizer1,
                       np.array(memory.observations),
                       np.array(memory.actions),
                       memory.rewards)
            
            train_step(model_p2, optimizer2,
                       np.array(memory2.observations),
                       np.array(memory2.actions),
                       memory2.rewards)
            
            # Save weights every 1000 episodes
            if episode % 1000 == 0:
                
                print(f"Saving weights for episode {episode}")
                path_p1 = f"models/weights/p1_weights_{episode}.h5"
                path_p2 = f"models/weights/p2_weights_{episode}.h5"
                model_p1.save_weights(path_p1)
                model_p2.save_weights(path_p2)
            break
        
## Show training log
games = list(train_log.keys())
wins = [train_log[game][0] for game in games]
win_p = [train_log[game][1] for game in games]


plt.plot(games, wins)
plt.xlabel('Games')
plt.ylabel('Wins')
plt.title('Training Wins')
plt.savefig('training_wins.png')

plt.plot(games, win_p)
plt.xlabel('Games')
plt.ylabel('Win Rate')
plt.title('Training Win Rate')
plt.savefig('training_winrate.png')

## Save model
model_p1.save_weights('models/DQN_weights_player1.h5')
model_p2.save_weights('models/DQN_weights_player2.h5')




