# Connect four rules
CONNECT_X = 4
BOARD_HEIGHT = 6
BOARD_WIDTH = 7

# Board
EMPTY = 0
YELLOW = 1
RED = 2

# PyTorch
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
