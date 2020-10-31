import torch
import torch.nn as nn

from constants import *

def b(c):
    with torch.no_grad():
        return g(c)

def g(c):
    return c + c

a = torch.tensor([1., 2.], requires_grad=True)
print(b(a))