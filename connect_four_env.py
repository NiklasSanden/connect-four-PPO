import torch
import numpy as np

from constants import *

class ConnectFourEnv():
    def __init__(self):
        self.board = np.ones((BOARD_HEIGHT, BOARD_WIDTH)) * EMPTY
        self.turn = YELLOW
    
    def step(self, action):
        row = self.takeAction(action)
        reward = -1.0
        done = True
        if row != -1:
            reward, done = self.getGameResult(action, row)
        return (self.formatState(), reward, done)

    def reset(self):
        self.board = np.ones((BOARD_HEIGHT, BOARD_WIDTH)) * EMPTY
        self.turn = YELLOW
        return self.formatState()

    def formatState(self):
        return torch.tensor([np.stack(((self.board == YELLOW).astype(np.float32), (self.board == RED).astype(np.float32)))], device=DEVICE)

    # Returns the row it's placed in or -1 if it was illegal
    def takeAction(self, action):
        row = -1
        for i in range(BOARD_HEIGHT - 1, -1, -1):
            if self.board[i][action] == 0:
                self.board[i][action] = self.turn
                row = i
                break
        self.turn = YELLOW if self.turn == RED else RED
        return row

    # Returns reward and done flag
    def getGameResult(self, lastAction, row):
        # Check for win
        if self.checkCountInDirection(row, lastAction, (1, 0)) or self.checkCountInDirection(row, lastAction, (0, 1)) or self.checkCountInDirection(row, lastAction, (1, 1)) or self.checkCountInDirection(row, lastAction, (-1, 1)):
            return (1.0, True)

        # Check for draw
        emptyCol = False
        for i in range(BOARD_WIDTH):
            if self.board[0][i] == EMPTY:
                emptyCol = True
                break

        return (0.0, not emptyCol)

    # Checks both positive and negative direction
    # Also checks for X in a row instead of four according to the constant: CONNECT_X
    def checkCountInDirection(self, row, col, dir):
        colour = self.board[row][col]
        count = 1
        currentPos = (row + dir[0], col + dir[1])
        while count < CONNECT_X and currentPos[0] >= 0 and currentPos[0] < BOARD_HEIGHT and currentPos[1] >= 0 and currentPos[1] < BOARD_WIDTH:
            if self.board[currentPos[0]][currentPos[1]] == colour:
                count += 1
                currentPos = (currentPos[0] + dir[0], currentPos[1] + dir[1])
            else:
                break
        
        currentPos = (row - dir[0], col - dir[1])
        while count < CONNECT_X and currentPos[0] >= 0 and currentPos[0] < BOARD_HEIGHT and currentPos[1] >= 0 and currentPos[1] < BOARD_WIDTH:
            if self.board[currentPos[0]][currentPos[1]] == colour:
                count += 1
                currentPos = (currentPos[0] - dir[0], currentPos[1] - dir[1])
            else:
                break
        
        return count >= CONNECT_X
