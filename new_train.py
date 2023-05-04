import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from Connect4 import Connect4
from Memory import Memory
from models.DQN import DQN

"""
train.py

Training loop for our DQN network. Code is borrowed heavily from
https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca,
with modifications to fit our game representation and network structure.
"""

# Pulled from TDS article
def compute_loss(logits, actions, rewards): 
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss

@tf.function
def train_step(model, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
      # Forward propagate through the agent network
        
        logits = model(observations)
        loss = compute_loss(logits, actions, rewards)
        grads = tape.gradient(loss, model.trainable_variables)
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
def get_action(model, observation, epsilon):
    #determine whether model action or random action based on epsilon
    act = np.random.choice(['model','random'], 1, p=[1-epsilon, epsilon])[0]
    observation = np.array(observation).reshape(1,6,7,1)
    logits = model.predict(observation, verbose=None)
    prob_weights = tf.nn.softmax(logits).numpy()
    
    if act == 'model':
        action = list(prob_weights[0]).index(max(prob_weights[0]))
    if act == 'random':
        action = np.random.choice([0, 1, 2, 3, 4, 5, 6])
        
    return action, prob_weights[0]


def random_turn(connect4):
    if len(connect4.available_moves()) == 0:
        return
    else:
        col = np.random.choice([0, 1, 2, 3, 4, 5, 6])
        connect4.drop_piece(col)

def get_adjacent(board, row, col, player):
    new_board = np.zeros((6, 7))
    for col in range(7):
        for row in range(6):
            ## Check horizontal
            ### Check left
            col2 = col-1
            while col2 >= max(0, col-3) and board[row][col2] == player:
                new_board[row][col] += 1
                col2 -= 1
            ### Check right
            col2 = col+1
            while col2 <= min(6, col+3) and board[row][col2] == player:
                new_board[row][col] += 1
                col2 += 1
            ## Check vertical
            ### Check up
            row2 = row-1
            while row2 >= max(0, row-3) and board[row2][col] == player:
                new_board[row][col] += 1
                row2 -= 1
            ### Check down
            row2 = row+1
            while row2 <= min(5, row+3) and board[row2][col] == player:
                new_board[row][col] += 1
                row2 += 1
            ## Check diagonal
            ### Check up-left
            row2 = row-1
            col2 = col-1
            while row2 >= max(0, row-3) and col2 >= max(0, col-3) and board[row2][col2] == player:
                new_board[row][col] += 1
                row2 -= 1
                col2 -= 1
            ### Check up-right
            row2 = row-1
            col2 = col+1
            while row2 >= max(0, row-3) and col2 <= min(6, col+3) and board[row2][col2] == player:
                new_board[row][col] += 1
                row2 -= 1
                col2 += 1
            ### Check down-left
            row2 = row+1
            col2 = col-1
            while row2 <= min(5, row+3) and col2 >= max(0, col-3) and board[row2][col2] == player:
                new_board[row][col] += 1
                row2 += 1
                col2 -= 1
            ### Check down-right
            row2 = row+1
            col2 = col+1
            while row2 <= min(5, row+3) and col2 <= min(6, col+3) and board[row2][col2] == player:
                new_board[row][col] += 1
                row2 += 1
                col2 += 1
    return new_board

def encode_board(board):
    new_board = np.empty([6,7,1], dtype=np.float64())
    encoding = {'e': 0, 'r': 1, 'y': 2}
    for row in range(6):
        for col in range(7):
            new_board[row][col] = encoding[board[row][col]]
    return new_board

def get_next_rows(board):
    next_row = [5, 5, 5, 5, 5, 5, 5]
    for col in range(7):
        for row in range(5, -1, -1):
            if board[row][col] == 'e':
                next_row[col] = row
                break
        if board[0][col] != 'e':
            next_row[col] = 0

    return next_row

### Training of model

env = Connect4()

### Instatiate models
model = DQN()
memory = Memory()
num_episodes = 2500
epsilon = 1

reward = 0
win_count = 0

optimizer = keras.optimizers.Adam(0.001)

train_log = {}

for episode in range(num_episodes):
    
    # Start new game
    env.new_board()
    observation = encode_board(env.board)
    memory.clear()
    
    epsilon = epsilon * .9998

    # Play game
    while True:
        action, _ = get_action(model, observation, epsilon)
        
        if env.check_tie():
            reward = 10
            print(str(episode) + ": " + str(memory.actions) + " - tie")
            done = [True, 'tie']
        else:
            done = env.check_win()
        
        # Set rewards for wins and losses
        if not done[0]:
            rows = get_next_rows(env.board)
            while rows[action] == -1:
                action, _ = get_action(model, observation, epsilon)
                reward -= 15
            else:
                reward = 1
        elif 'y' in done[1]: # random player wins
            reward = -100
            print(str(episode) + ": " + str(memory.actions) + " - random wins")
        elif 'r' in done[1]: # network wins
            win_count += 1
            reward = 100
            print(str(episode) + ": " + str(memory.actions) + " - network wins")
            # env.print_board()

        observation = encode_board(env.board)

        memory.add_to_memory(observation, action, reward)
        
        random_turn(env)

        if done[0]:
            
            train_log[episode] = (win_count, win_count / (episode+1))
            
            # train network
            train_step(model, optimizer,
                       np.array(memory.observations),
                       np.array(memory.actions),
                       memory.rewards)
            break
        
## Show training log
games = list(train_log.keys())
wins = [train_log[game][0] for game in games]
win_p = [train_log[game][1] for game in games]


plt.plot(games, wins)
plt.xlabel('Games')
plt.ylabel('Wins')
plt.title('Training Wins')
plt.show()

plt.plot(games, win_p)
plt.xlabel('Games')
plt.ylabel('Win Rate')
plt.title('Training Win Rate')
plt.show()

## Save model
model.save_weights('models/DQN_weights.h5')




