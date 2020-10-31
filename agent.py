import torch
import torch.nn as nn

import os
import math

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

class PPOFull(nn.Module):
    def __init__(self, n_outputs):
        super(PPOFull, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, CONNECT_X),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * (BOARD_HEIGHT - CONNECT_X + 1) * (BOARD_WIDTH - CONNECT_X + 1), 512),
            nn.ReLU(),
            nn.Linear(512, n_outputs)
        )

    def forward(self, X):
        assert len(X.shape) == 4 and X.shape[1] == 2 and X.shape[2] == BOARD_HEIGHT and X.shape[3] == BOARD_WIDTH
        out = self.cnn(X).view(X.shape[0], -1)
        return self.fc(out)

class Agent():
    def __init__(self, name):
        self.name = name
        
        # Use if you want a them to have the same base - remember to uncomment it in saving and loading
        #self.base = PPOBase().to(DEVICE)
        #self.V = PPOHead(self.base, 1).to(DEVICE)
        #self.PI = PPOHead(self.base, BOARD_WIDTH).to(DEVICE)
        #self.optimizer = torch.optim.Adam(list(self.base.parameters()) + list(self.V.parameters()) + list(self.PI.parameters()), lr=LEARNING_RATE)
        
        self.V = PPOFull(1).to(DEVICE)
        self.PI = PPOFull(BOARD_WIDTH).to(DEVICE)
        self.optimizer = torch.optim.SGD(list(self.V.parameters()) + list(self.PI.parameters()), lr=LEARNING_RATE)

        self.load()
    
    def load(self):
        if os.path.isdir("save"):
            #self.base.load_state_dict(torch.load("save/" + self.name + "_base.pt", map_location=DEVICE))
            self.V.load_state_dict(torch.load("save/" + self.name + "_V.pt", map_location=DEVICE))
            self.PI.load_state_dict(torch.load("save/" + self.name + "_PI.pt", map_location=DEVICE))
            self.optimizer.load_state_dict(torch.load("save/" + self.name + "_optimizer.pt", map_location=DEVICE))
            print("Agent ", self.name, " loaded")
        else:
            print("Couldn't find a saved checkpoint for agent ", self.name)

    def save(self):
        if os.path.isdir("save") == False:
            os.mkdir("save")
        #torch.save(self.base.state_dict(), "save/" + self.name + "_base.pt")
        torch.save(self.V.state_dict(), "save/" + self.name + "_V.pt")
        torch.save(self.PI.state_dict(), "save/" + self.name + "_PI.pt")
        torch.save(self.optimizer.state_dict(), "save/" + self.name + "_optimizer.pt")
        print("Agent ", self.name, " saved")

    #################################################################################
    
    def calc_entropy(self, probs):
        entropy = -torch.sum(probs * torch.log(probs), dim=1, keepdim=True)
        assert (entropy < 0.0).sum() == 0 # Make sure entropy is non-negative
        assert len(entropy.shape) == 2 and entropy.shape[1] == 1
        return entropy.view(-1)

    def get_action_probs(self, states):
        logits = self.PI(states)
        assert len(logits.shape) == 2 and logits.shape[0] == states.shape[0] and logits.shape[1] == BOARD_WIDTH
        m, _ = torch.max(logits, dim=1, keepdim=True)
        assert len(m.shape) == 2 and m.shape[0] == logits.shape[0] and m.shape[1] == 1
        logits -= m
        probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)
        probs = torch.clamp(probs, 0.001, 0.999)
        assert len(probs.shape) == 2 and probs.shape[0] == states.shape[0] and probs.shape[1] == BOARD_WIDTH
        return probs

    def get_action_probs_no_grad(self, states):
        with torch.no_grad():
            return self.get_action_probs(states)

    def choose_actions(self, states):
        probs = self.get_action_probs_no_grad(states)
        actions = torch.multinomial(probs, 1)
        assert len(actions.shape) == 2 and actions.shape[0] == states.shape[0] and actions.shape[1] == 1
        return actions.view(-1)
    
    def get_state_values(self, states):
        values = self.V(states)
        assert len(values.shape) == 2 and values.shape[0] == states.shape[0] and values.shape[1] == 1
        return values

    def get_state_values_no_grad(self, states):
        with torch.no_grad():
            return self.get_state_values(states)
    
    #################################################################################

    def start_training_epoch(self):
        self.V.train()
        self.PI.train()

        self.trajectory_states = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
    
    def add_state_action_reward(self, state, action, reward, traj_id):
        if len(self.trajectory_states) <= traj_id:
            self.trajectory_states.append([state])
            self.trajectory_actions.append([torch.tensor([action], dtype=torch.long, device=DEVICE)])
            self.trajectory_rewards.append([torch.tensor([reward], dtype=torch.float32, device=DEVICE)])
        else:
            self.trajectory_states[traj_id].append(state)
            self.trajectory_actions[traj_id].append(torch.tensor([action], dtype=torch.torch.long, device=DEVICE))
            self.trajectory_rewards[traj_id].append(torch.tensor([reward], dtype=torch.float32, device=DEVICE))

    def replace_last_reward(self, new_reward, traj_id):
        assert len(self.trajectory_rewards[traj_id]) > 0
        self.trajectory_rewards[traj_id][-1] = torch.tensor([new_reward], dtype=torch.float32, device=DEVICE)

    #################################################################################

    def compute_advantage(self):
        # Turn each trajectory into one tensor
        for traj in range(len(self.trajectory_states)):
            self.trajectory_states[traj] = torch.cat(self.trajectory_states[traj])
            self.trajectory_actions[traj] = torch.cat(self.trajectory_actions[traj])
            self.trajectory_rewards[traj] = torch.cat(self.trajectory_rewards[traj])

        # Compute bellman residuals
        self.trajectory_bellman = []
        for traj in range(len(self.trajectory_rewards)):
            state_values = self.get_state_values_no_grad(self.trajectory_states[traj]).view(-1)
            state_values = torch.cat((state_values, torch.tensor([0.0], device=DEVICE))) # Add terminal state
            self.trajectory_bellman.append([torch.tensor([self.trajectory_rewards[traj][0] + GAMMA * state_values[1] - state_values[0]], device=DEVICE)])
            
            index = 1
            for reward in self.trajectory_rewards[traj][1:]:
                new_bellman = torch.tensor([reward + GAMMA * state_values[index + 1] - state_values[index]], device=DEVICE)
                self.trajectory_bellman[traj].append(new_bellman)
                index += 1
            
            # Turn each trajectory into one tensor
            self.trajectory_bellman[traj] = torch.cat(self.trajectory_bellman[traj])
        
        assert len(self.trajectory_bellman) == len(self.trajectory_states)
        assert len(self.trajectory_bellman[0].shape) == 1 and self.trajectory_bellman[0].shape[0] == self.trajectory_rewards[0].shape[0]

        # Compute advantage
        self.trajectory_advantage = []
        for traj in range(len(self.trajectory_bellman)):
            self.trajectory_advantage.append([torch.tensor([self.trajectory_bellman[traj][-1]], device=DEVICE)])
            for bellman in reversed(self.trajectory_bellman[traj][:-1]):
                new_advantage = torch.tensor([bellman + GAE_LAMBDA * GAMMA * self.trajectory_advantage[traj][-1]], device=DEVICE)
                self.trajectory_advantage[traj].append(new_advantage)
            
            # Turn each trajectory into one tensor
            self.trajectory_advantage[traj].reverse()
            self.trajectory_advantage[traj] = torch.cat(self.trajectory_advantage[traj])
        
        assert len(self.trajectory_advantage) == len(self.trajectory_states)
        assert len(self.trajectory_advantage[0].shape) == 1 and self.trajectory_advantage[0].shape[0] == self.trajectory_rewards[0].shape[0]

        # Compute returns (used to update V)
        self.trajectory_returns = []
        for traj in range(len(self.trajectory_rewards)):
            self.trajectory_returns.append([torch.tensor([self.trajectory_rewards[traj][-1]], device=DEVICE)])
            for reward in reversed(self.trajectory_rewards[traj][:-1]):
                new_reward = torch.tensor([reward + GAMMA * self.trajectory_returns[traj][-1]], device=DEVICE)
                self.trajectory_returns[traj].append(new_reward)
            
            # Turn each trajectory into one tensor
            self.trajectory_returns[traj].reverse()
            self.trajectory_returns[traj] = torch.cat(self.trajectory_returns[traj])
        
        assert len(self.trajectory_returns) == len(self.trajectory_states)
        assert len(self.trajectory_returns[0].shape) == 1 and self.trajectory_returns[0].shape[0] == self.trajectory_rewards[0].shape[0]
        
        # Concatenate trajectories
        self.trajectory_states = torch.cat(self.trajectory_states)
        self.trajectory_actions = torch.cat(self.trajectory_actions)
        self.trajectory_advantage = torch.cat(self.trajectory_advantage)
        self.trajectory_returns = torch.cat(self.trajectory_returns)

        assert len(self.trajectory_states.shape) > 1
        assert len(self.trajectory_actions.shape) == 1
        assert len(self.trajectory_advantage.shape) == 1
        assert len(self.trajectory_returns.shape) == 1

    #################################################################################

    def train(self):
        # Get old probs
        old_probs = self.get_action_probs_no_grad(self.trajectory_states) # old action probabilities for all possible actions
        old_action_probs = old_probs[torch.arange(old_probs.shape[0]), self.trajectory_actions] # old action probabilities for the actions chosen
        assert len(old_action_probs.shape) == 1 and old_action_probs.shape[0] == old_probs.shape[0]

        # Dataset
        dataset = torch.utils.data.TensorDataset(self.trajectory_states, self.trajectory_actions, self.trajectory_advantage, self.trajectory_returns, old_probs, old_action_probs)
        dataloader = torch.utils.data.DataLoader(dataset, min(BATCH_SIZE, len(self.trajectory_states)), shuffle=True)

        # Update PI and V
        outside_KL = False
        for i in range(SGD_EPOCHS):
            self.optimizer.zero_grad()
            
            for traj_states, traj_actions, traj_advantage, traj_returns, old_probs, old_action_probs in dataloader:
                # Calculate PI score (negated later to get PI loss)
                if not outside_KL:
                    probs = self.get_action_probs(traj_states)
                    entropy = self.calc_entropy(probs)
                    action_probs = probs[torch.arange(probs.shape[0]), traj_actions]
                    assert len(action_probs.shape) == 1 and action_probs.shape[0] == probs.shape[0]

                    ratio = action_probs / old_action_probs
                    assert ratio.shape == traj_advantage.shape
                    surr1 = ratio * traj_advantage
                    surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON) * traj_advantage
                    assert surr1.shape == surr2.shape

                    PI_score = (torch.min(surr1, surr2) + ENTROPY_FACTOR * entropy).mean()
                else:
                    PI_score = torch.tensor([0.0], device=DEVICE)

                # Calculate V loss
                state_value_predictions = self.get_state_values(traj_states).view(-1)
                assert state_value_predictions.shape == traj_returns.shape
                V_loss = (traj_returns - state_value_predictions).pow(2).mean()

                # Backpropogate
                loss = (V_loss - PI_TO_V_IMPORTANCE_FACTOR * PI_score) / math.ceil(len(self.trajectory_states) / min(BATCH_SIZE, len(self.trajectory_states)))
                loss.backward()

                # Check KL-divergence for early termination of PI loss
                if not outside_KL:
                    with torch.no_grad():
                        kl_divergence = torch.sum(probs * torch.log(probs / old_probs), dim=1).mean()
                        if kl_divergence >= MAX_KL_DIVERGENCE:
                            outside_KL = True
                            #print("Outside_KL: ", self.name)
            
            # Optimize
            self.optimizer.step()

class AgentsTrainer():
    def __init__(self, env):
        self.env = env
        self.firstAgent = Agent("first")
        self.secondAgent = Agent("second")
    
    def train(self, epochs):
        for epoch in range(epochs):
            print("Epoch: ", epoch + 1)

            self.firstAgent.start_training_epoch()
            self.secondAgent.start_training_epoch()

            # Gather experience
            for traj in range(NUM_TRAJECTORIES):
                # Reset environment
                state = self.env.reset()
                agents = [self.firstAgent, self.secondAgent]
                turn = 0

                done = False
                while not done:
                    action = agents[turn].choose_actions(state)
                    next_state, reward, done = self.env.step(action.cpu().item())
                    assert len(next_state.shape) == 4

                    agents[turn].add_state_action_reward(state, action, reward, traj)

                    state = next_state
                    turn = 1 - turn

                    if done:
                        agents[turn].replace_last_reward(-reward, traj)
            
            self.firstAgent.compute_advantage()
            self.firstAgent.train()
            self.secondAgent.compute_advantage()
            self.secondAgent.train()

            if ((epoch + 1) % SAVE_EVERY_X_EPOCHS == 0 or epoch + 1 == epochs):
                self.firstAgent.save()
                self.secondAgent.save()
    
    def get_action(self, state, first_agent=True):
        if first_agent:
            probs = self.firstAgent.get_action_probs_no_grad(state)
            action = self.firstAgent.choose_actions(state).cpu().item()
        else:
            probs = self.secondAgent.get_action_probs_no_grad(state)
            action = self.secondAgent.choose_actions(state).cpu().item()
        
        print("Probs: ", probs)
        return action
