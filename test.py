import torch
import torch.nn as nn

from constants import *

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.a = 25
        self.f = nn.Linear(2, 2)

a = Test()
sd = a.state_dict()
a.a = 2
a.load_state_dict(sd)
print(a.a)