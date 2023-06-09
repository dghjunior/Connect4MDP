#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory.py

A Memory class representing the memory of the DQN network. Taken from 
https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca.

"""

class Memory:
    def __init__(self): 
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self): 
        self.observations = []
        self.actions = []
        self.rewards = []
        self.info = []
        
    def add_to_memory(self, new_observation, new_action, new_reward): 
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(float(new_reward))