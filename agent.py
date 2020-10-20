import torch
import torch.nn as nn

import os

from constants import *

class PPOBase(nn.Module):
    def __init__(self):
        super(PPOBase, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, CONNECT_X),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, X):
        assert len(X.shape) == 4 and X.shape[1] == 2 and X.shape[2] == BOARD_HEIGHT and X.shape[3] == BOARD_WIDTH
        return self.cnn(X).view(X.shape[0], -1)

class PPOHead(nn.Module):
    def __init__(self, base, n_outputs):
        super(PPOHead, self).__init__()

        self.base = [base] # Put it in a list so that it does not get auto-registered as a parameter

        self.fc = nn.Sequential(
            nn.Linear(64 * (BOARD_HEIGHT - CONNECT_X + 1) * (BOARD_WIDTH - CONNECT_X + 1), 512),
            nn.ReLU(),
            nn.Linear(512, n_outputs)
        )
    
    def forward(self, X):
        return self.fc(self.base[0](X))

class Agent():
    def __init__(self, name):
        self.name = name
        self.base = PPOBase().to(DEVICE)
        self.V = PPOHead(self.networkBase, 1).to(DEVICE)
        self.PI = PPOHead(self.networkBase, BOARD_WIDTH).to(DEVICE)
        self.optimizer = torch.optim.Adam(list(self.networkBase.parameters()) + list(self.V.parameters()) + list(self.PI.parameters()))
        self.load()
    
    def load(self):
        if os.path.isdir("save"):
            self.base.load_state_dict(torch.load("save/" + name + "_base.pt", map_location=DEVICE))
            self.V.load_state_dict(torch.load("save/" + name + "_V.pt", map_location=DEVICE))
            self.PI.load_state_dict(torch.load("save/" + name + "_PI.pt", map_location=DEVICE))
            self.optimizer.load_state_dict(torch.load("save/" + name + "_optimizer.pt", map_location=DEVICE))
    
    def save(self):
        if os.path.isidr("save") == False:
            os.mkdir("save")
        torch.save(self.base.state_dict(), "save/" + name + "_base.pt")
        torch.save(self.V.state_dict(), "save/" + name + "_V.pt")
        torch.save(self.PI.state_dict(), "save/" + name + "_PI.pt")
        torch.save(self.optimizer.state_dict(), "save/" + name + "_optimizer.pt")

    def get_action_probs_no_grad(self, states):
        with torch.no_grad():
            logits = self.PI(states)
            assert len(logits.shape) == 2 and logits.shape[0] == states.shape[0] and logits.shape[1] == BOARD_WIDTH
            m, _ = torch.max(logits, dim=1, keepdim=True)
            assert len(m.shape) == 2 and m.shape[0] == logits.shape[0] and m.shape[1] == 1
            logits -= m
            probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)
        assert len(probs.shape) == 2 and probs.shape[0] == states.shape[0] and probs.shape[1] == BOARD_WIDTH
        return probs
    
    def get_action_probs(self, states):
        logits = self.PI(states)
        assert len(logits.shape) == 2 and logits.shape[0] == states.shape[0] and logits.shape[1] == BOARD_WIDTH
        m, _ = torch.max(logits, dim=1, keepdim=True)
        assert len(m.shape) == 2 and m.shape[0] == logits.shape[0] and m.shape[1] == 1
        logits -= m
        probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)
        assert len(probs.shape) == 2 and probs.shape[0] == states.shape[0] and probs.shape[1] == BOARD_WIDTH
        return probs

    def choose_actions(self, states):
        probs = self.get_action_probs_no_grad(states)
        actions = torch.multinomial(probs, 1)
        assert len(actions.shape) == 2 and actions.shape[0] == states.shape[0] and actions.shape[1] == 1
        return actions.view(-1)
    
    def get_state_values_no_grad(self, states):
        with torch.no_grad():
            values = self.V(states)
        assert len(values.shape) == 2 and values.shape[0] == states.shape[0] and values.shape[1] == 1
        return values
    
    def get_state_values(self, states):
        values = self.V(states)
        assert len(values.shape) == 2 and values.shape[0] == states.shape[0] and values.shape[1] == 1
        return values


class AgentsTrainer():
    def __init__(self):
        self.firstAgent = Agent("first")
        self.secondAgent = Agent("second")
