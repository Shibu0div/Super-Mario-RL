import time 
class MetricLogger:
    def __init__(self,save_dir):
        self.save_log = save_dir / "log" 
        with open(self.save_log,"w") as f:
            f.write("Episode,EpisodeReward,Epsilon,BestReward\n")
        self.best_reward = float("-inf")
    def log_episode(self,episode,episode_reward,epsilon):
        self.best_reward = max(self.best_reward,episode_reward) 
        print(
            f"Episode: {episode} | "
            f"Reward: {episode_reward:.2f} |"
            f"Epsilon: {epsilon:.3f} |"
            f"Best Reward: {self.best_reward:.2f}"
        )
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode},{episode_reward:.2f},"
                f"{epsilon:.3f},{self.best_reward:.2f}\n"
            )