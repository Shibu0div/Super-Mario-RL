import yaml
import torch
from dqn import dqn
import numpy as np
import torch.optim as optim
from torch.nn import SmoothL1Loss 
from memory import Memory
class Mario:
    def __init__(self,state_dim,action_dim,save_dir):
        with open("parameters.yaml",'r') as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set["supermariov0"]
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.save_dir = save_dir
        self.memory = Memory()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = dqn(self.state_dim,self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = params["exploration_rate"]
        self.exploration_rate_decay = params["exploration_rate_decay"]
        self.exploration_rate_min = params["exploration_rate_min"]
        self.gamma = params["gamma"]
        self.batch_size = params["batch_size"]
        self.burnin = params["burnin"]
        self.learn_every = params["learn_every"]
        self.sync_every = params["sync_every"]
        self.curr_step = 0

        self.save_every = params["save_every"]
        self.optimizer = optim.Adam(self.net.parameters(),lr=0.00025)
        self.loss_fn = SmoothL1Loss()
        
    def act(self,state):
        # Explore
        if np.random.rand() < self.exploration_rate :
            action_idx = np.random.randint(self.action_dim) 
        else: # Exploit
            state = state[0].__array__() if isinstance(state,tuple) else state.__array__()
            state = torch.tensor(state,device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, dim=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step +=1 
        return action_idx
    def td_estimate(self,state,action):
        current_Q = self.net(state,model="online")[
            np.arange(0,self.batch_size),action
        ]
        return current_Q
    @torch.no_grad()
    def td_target(self,reward,next_state,done):
        next_state_Q = self.net(next_state,model="online")
        best_action = torch.argmax(next_state_Q,dim=1)
        next_Q = self.net(next_state,model="target")[
            np.arange(0,self.batch_size), best_action
        ]
        return (reward+(1-done.float()) * self.gamma * next_Q).float()
    def update_Q_online(self,td_estimate, td_target):
        loss = self.loss_fn(td_estimate,td_target)
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step() 
        return loss.item()
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step//self.save_every)}.pt"
        
        torch.save(self.net.state_dict(),save_path)
        print(f"Mario DQN saved to {save_path} at step {self.curr_step}")
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        
        if self.curr_step < self.burnin:
            return None, None 
        if self.curr_step % self.learn_every != 0:
            return None,None 
        state, next_state, action, reward, done = self.memory.recall()
        td_est = self.td_estimate(state,action) 
        td_tgt = self.td_target(reward,next_state,done) 
        loss = self.update_Q_online(td_est,td_tgt)
        return (td_est.mean().item(),loss)

