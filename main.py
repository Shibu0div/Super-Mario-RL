import gym 
from gym.spaces import Discrete
import gym_super_mario_bros 
from nes_py.wrappers import JoypadSpace
from preprocess_env import SkipFrame, GrayScaleObservation, ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from pathlib import Path 
import datetime
from agent import Mario
from metric_logger import MetricLogger

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0",render_mode="rgb_array")
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

env = SkipFrame(env,skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env,shape=84) 
env = FrameStack(env,num_stack=4)

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)


action_space = env.action_space
if hasattr(action_space,"n"):
    action_dim = action_space.n 
else:
    raise ValueError(f"Expected discrete action space with attribute 'n', got {type(action_space)}")

mario = Mario(state_dim=(4,84,84), action_dim=action_dim, save_dir=save_dir)
logger = MetricLogger(save_dir) 

episodes = 40000 
best_reward = float("-inf")
for e in range(episodes):
    state = env.reset() 

    while True:
        action = mario.act(state)

        next_state, reward, done, trunc, info = env.step(action) 

        mario.memory.cache(state,next_state,action,reward,done) 

        q, loss = mario.learn()

        state = next_state
        if best_reward < reward:
            best_reward = reward
            mario.save()
        
        if done or info["flag_get"]:
            break 
    logger.log_episode(e,reward,mario.exploration_rate)
    
        

        


