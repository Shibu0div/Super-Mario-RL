from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Memory:
    def __init__(self):
        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(100000,device=torch.device("cpu")))
        self.batch_size = 32
    def cache(self, state, next_state, action, reward,done):
        def first_if_tuple(x):
            return x[0] if isinstance(x,tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state) 
        next_state = torch.tensor(next_state) 
        action = torch.tensor([action]) 
        reward = torch.tensor([reward]) 
        done = torch.tensor([done]) 
        self.memory.add(TensorDict({"state":state,"next_state":next_state,"action":action,"reward":reward,"done":done},batch_size=[]))
    def recall(self):
        batch = self.memory.sample(self.batch_size).to(device)
        state = batch.get("state")
        next_state = batch.get("next_state")
        action = batch.get("action")
        reward = batch.get("reward")
        done = batch.get("done")
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
