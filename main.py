import shutil
import os

import torch

from connect_four_env import ConnectFourEnv
from agent import AgentsTrainer
from constants import *

def playAgent(env, agentsTrainer):
    while True:
        yourTurn = int(input("Go first? (1=Yes, 0=No, 2=Quit) "))
        if yourTurn == 2:
            break

        playerStarted = yourTurn == 1
        
        state = env.reset()
        done = False
        print(env.board)
        while not done:
            action = 0
            if yourTurn == 1:
                action = int(input("Action: ")) - 1
            else:
                if playerStarted:
                    action = agentsTrainer.get_action(state, False)
                else:
                    action = agentsTrainer.get_action(state, True)
            state, reward, done = env.step(action)
            print(env.board)
            yourTurn = 1 - yourTurn

if __name__ == '__main__':
    env = ConnectFourEnv()
    agentsTrainer = AgentsTrainer(env)
    
    train = int(input("Train? (1=Yes, 0=No) "))
    if train == 1:
        agentsTrainer.train(TRAINING_EPOCHS)

    play = int(input("Play? (1=Yes, 0=No) "))
    if play == 1:
        playAgent(env, agentsTrainer)
    
    print("All done")