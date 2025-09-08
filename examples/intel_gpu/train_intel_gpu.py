#!/usr/bin/env python3
"""
Simple training example for LeRobot on Intel GPU.
This script demonstrates how to train a policy using Intel GPU acceleration.
"""
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import LeRobot utilities
from lerobot.utils.utils import get_safe_torch_device, is_amp_available
from lerobot.utils.device_utils import synchronize_device, empty_cache, get_memory_info

@dataclass
class TrainingConfig:
    """Training configuration for Intel GPU."""
    device: str = "xpu"
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_episodes: int = 1000
    hidden_size: int = 128
    gamma: float = 0.99
    use_amp: bool = True
    log_interval: int = 10
    save_interval: int = 100

class PolicyNetwork(nn.Module):
    """Policy network optimized for Intel GPU."""
    
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()
    
    def forward(self, x):
        return self.network(x)
    
    def get_action(self, obs, deterministic=False):
        """Get action from observation."""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
            
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            
            probs = self.forward(obs)
            
            if deterministic:
                return torch.argmax(probs, dim=-1)
            else:
                return torch.multinomial(probs, 1).squeeze(-1)

class IntelGPUTrainer:
    """Trainer class optimized for Intel GPU."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = get_safe_torch_device(config.device, log=True)
        
        # Setup environment
        self.env = gym.make("CartPole-v1")
        obs_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        # Setup policy
        self.policy = PolicyNetwork(obs_size, action_size, config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Setup gradient scaler for mixed precision
        self.scaler = torch.cuda.GradScaler() if config.use_amp and self.device.type == "cuda" else None
        if config.use_amp and self.device.type == "xpu":
            # Intel GPU scaler (if available)
            try:
                self.scaler = torch.xpu.amp.GradScaler()
            except:
                self.scaler = None
                logger.warning("Intel GPU GradScaler not available, using regular training")
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def collect_episode(self):
        """Collect a single episode of experience."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        while True:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            # Get action
            with torch.autocast(device_type=self.device.type, enabled=self.config.use_amp):
                action_probs = self.policy(obs_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            # Store experience
            states.append(obs.copy())
            actions.append(action.item())
            log_probs.append(log_prob)
            
            # Take step
            obs, reward, terminated, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        return {
            'states': np.array(states),
            'actions': actions,
            'rewards': rewards,
            'log_probs': torch.stack(log_probs),
            'episode_reward': episode_reward,
            'episode_length': episode_length
        }
    
    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.config.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update_policy(self, experience):
        """Update policy using collected experience."""
        states = torch.FloatTensor(experience['states']).to(self.device)
        log_probs = experience['log_probs'].to(self.device)
        returns = self.compute_returns(experience['rewards'])
        
        # Compute policy loss
        loss = -(log_probs * returns).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        
        if self.scaler:
            # Mixed precision training
            with torch.autocast(device_type=self.device.type):
                loss_scaled = self.scaler.scale(loss)
            
            loss_scaled.backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.num_episodes} episodes")
        
        for episode in range(self.config.num_episodes):
            # Collect experience
            experience = self.collect_episode()
            
            # Update policy
            loss = self.update_policy(experience)
            
            # Store metrics
            self.episode_rewards.append(experience['episode_reward'])
            self.episode_lengths.append(experience['episode_length'])
            self.losses.append(loss)
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.config.log_interval:])
                avg_loss = np.mean(self.losses[-self.config.log_interval:])
                
                # Get memory info
                memory_info = get_memory_info(self.device)
                memory_mb = memory_info.get('allocated', 0) / 1024**2
                
                logger.info(
                    f"Episode {episode + 1:4d} | "
                    f"Avg Reward: {avg_reward:6.2f} | "
                    f"Avg Length: {avg_length:5.1f} | "
                    f"Loss: {avg_loss:6.4f} | "
                    f"Memory: {memory_mb:5.1f}MB"
                )
            
            # Save checkpoint
            if (episode + 1) % self.config.save_interval == 0:
                self.save_checkpoint(episode + 1)
            
            # Clear cache periodically
            if (episode + 1) % 50 == 0:
                empty_cache(self.device)
                synchronize_device(self.device)
        
        logger.info("Training completed!")
        self.save_checkpoint("final")
    
    def save_checkpoint(self, episode):
        """Save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'config': self.config
        }
        
        checkpoint_path = f"intel_gpu_policy_episode_{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def evaluate(self, num_episodes=10):
        """Evaluate the trained policy."""
        logger.info(f"Evaluating policy for {num_episodes} episodes")
        
        self.policy.eval()
        eval_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                action = self.policy.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            logger.info(f"Eval Episode {episode + 1}: Reward = {episode_reward}")
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"Best Reward: {np.max(eval_rewards):.2f}")
        
        self.policy.train()
        return eval_rewards
    
    def cleanup(self):
        """Clean up resources."""
        self.env.close()
        empty_cache(self.device)

def main():
    """Main function."""
    print("🚀 LeRobot Intel GPU Training Example\n")
    
    # Load configuration
    config = TrainingConfig()
    
    # Create trainer
    trainer = IntelGPUTrainer(config)
    
    try:
        # Train the policy
        trainer.train()
        
        # Evaluate the policy
        trainer.evaluate(num_episodes=5)
        
        # Print summary
        print("\n" + "="*50)
        print("📊 Training Summary")
        print("="*50)
        print(f"Total episodes: {len(trainer.episode_rewards)}")
        print(f"Average reward: {np.mean(trainer.episode_rewards):.2f}")
        print(f"Best reward: {np.max(trainer.episode_rewards):.2f}")
        print(f"Final 100 episodes average: {np.mean(trainer.episode_rewards[-100:]):.2f}")
        print(f"Device used: {trainer.device}")
        
        # Memory info
        memory_info = get_memory_info(trainer.device)
        if memory_info.get('allocated', 0) > 0:
            print(f"Final memory usage: {memory_info['allocated'] / 1024**2:.1f} MB")
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()
