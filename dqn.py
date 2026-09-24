import torch 
import torch.nn as nn

class dqn(nn.Module):
    def __int__(self,input_dim,output_dim):
        super(dqn,self).__init__()
        c,h,w = input_dim
         
        self.online = nn.Sequential(
            nn.Conv2d(c,32,kernel_size=8,stride=4), 
            nn.ReLU(), 
            nn.Conv2d(32,64,kernel_size=4,stride=2), 
            nn.ReLU(), 
            nn.Conv2d(64,64,kernel_size=3,stride=1), 
            nn.ReLU(), 
            nn.Flatten(), 
            nn.Linear(3136,512), 
            nn.ReLU(), 
            nn.Linear(512,output_dim),
        ) # This is the network that is actively learning.
        self.target = nn.Sequential(
            nn.Conv2d(c,32,kernel_size=8,stride=4), 
            nn.ReLU(), 
            nn.Conv2d(32,64,kernel_size=4,stride=2), 
            nn.ReLU(), 
            nn.Conv2d(64,64,kernel_size=3,stride=1), 
            nn.ReLU(), 
            nn.Flatten(), 
            nn.Linear(3136,512), 
            nn.ReLU(), 
            nn.Linear(512,output_dim),
        ) # This is a second, more stable copy of the network.
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters(): 
            p.requires_grad = False 
        def forward(self,input,model): 
            if model=="online": 
                return self.online(input)
            elif model=="target": 
                return self.target(input)
        
