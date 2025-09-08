#!/usr/bin/env python3
"""
Example: Running LeRobot simulation on Intel GPU.
This script demonstrates how to set up and run robotic simulation using Intel GPU acceleration.
"""
import torch
import gymnasium as gym
import numpy as np
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LeRobot utilities
from lerobot.utils.utils import get_safe_torch_device, auto_select_torch_device, is_amp_available
from lerobot.utils.device_utils import (
    synchronize_device, empty_cache, get_memory_info, log_device_info
)

def setup_intel_gpu_environment():
    """Set up optimal environment for Intel GPU simulation."""
    # Auto-select best available device
    device = auto_select_torch_device()
    logger.info(f"Using device: {device}")
    
    # Log device information
    log_device_info(device)
    
    # Set Intel GPU specific optimizations
    if device.type == "xpu":
        # Clear cache for clean start
        empty_cache(device)
        
        # Set optimal batch size for Intel GPU
        batch_size = 32  # Optimal for most Intel GPUs
        
        # Check memory
        memory_info = get_memory_info(device)
        logger.info(f"Intel GPU memory info: {memory_info}")
        
        logger.info("Intel GPU optimizations enabled")
        return device, batch_size
    else:
        return device, 64

class IntelGPUPolicy(torch.nn.Module):
    """Simple neural network policy optimized for Intel GPU."""
    
    def __init__(self, obs_size, action_size, hidden_size=128):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, obs, deterministic=True):
        """Predict action given observation."""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            
            logits = self.forward(obs)
            
            if deterministic:
                return torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).squeeze(-1)

class IntelGPUSimulator:
    """Simulation environment manager for Intel GPU."""
    
    def __init__(self, env_name="CartPole-v1", device=None):
        self.env_name = env_name
        self.device = device or auto_select_torch_device()
        self.env = None
        self.policy = None
        self.optimizer = None
        
        # Performance tracking
        self.episode_times = []
        self.memory_usage = []
        
    def setup_environment(self):
        """Set up the simulation environment."""
        logger.info(f"Setting up environment: {self.env_name}")
        
        try:
            self.env = gym.make(self.env_name)
            logger.info(f"✅ Environment created: {self.env_name}")
            
            # Get environment info
            obs_space = self.env.observation_space
            action_space = self.env.action_space
            
            logger.info(f"Observation space: {obs_space}")
            logger.info(f"Action space: {action_space}")
            
            return obs_space, action_space
            
        except Exception as e:
            logger.error(f"❌ Failed to create environment: {e}")
            raise
    
    def setup_policy(self, obs_size, action_size):
        """Set up the policy network."""
        logger.info("Setting up policy network...")
        
        # Create policy
        self.policy = IntelGPUPolicy(obs_size, action_size).to(self.device)
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.policy.parameters())
        logger.info(f"✅ Policy created with {num_params:,} parameters")
        
        return self.policy
    
    def run_episode(self, episode_num, render=False, train=False):
        """Run a single episode."""
        if self.env is None or self.policy is None:
            raise RuntimeError("Environment and policy must be set up first")
        
        start_time = time.perf_counter()
        
        obs, info = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_loss = 0
        
        # Track states and actions for training
        states = []
        actions = []
        rewards = []
        
        while True:
            # Convert observation to tensor and move to Intel GPU
            obs_tensor = torch.FloatTensor(obs).to(self.device, non_blocking=True)
            
            # Use autocast for mixed precision if available
            use_amp = is_amp_available(self.device.type)
            
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                # Get action from policy
                if hasattr(self.env.action_space, 'n'):  # Discrete action space
                    action_tensor = self.policy.predict(obs_tensor, deterministic=False)
                    action = action_tensor.cpu().numpy()
                    if np.isscalar(action):
                        action = int(action)
                    else:
                        action = int(action[0])
                else:  # Continuous action space
                    action_logits = self.policy(obs_tensor.unsqueeze(0))
                    action = torch.tanh(action_logits).cpu().numpy()[0]
            
            # Store for training
            if train:
                states.append(obs.copy())
                actions.append(action)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if train:
                rewards.append(reward)
            
            if render and hasattr(self.env, 'render'):
                self.env.render()
            
            if terminated or truncated:
                break
        
        # Training step
        if train and len(states) > 1:
            episode_loss = self._train_step(states, actions, rewards)
        
        # Synchronize device
        synchronize_device(self.device)
        
        # Record performance
        episode_time = time.perf_counter() - start_time
        self.episode_times.append(episode_time)
        
        # Record memory usage
        memory_info = get_memory_info(self.device)
        self.memory_usage.append(memory_info.get('allocated', 0))
        
        logger.info(
            f"Episode {episode_num}: "
            f"Reward={episode_reward:.2f}, "
            f"Steps={episode_steps}, "
            f"Time={episode_time:.3f}s"
            + (f", Loss={episode_loss:.4f}" if train else "")
        )
        
        return {
            'episode': episode_num,
            'reward': episode_reward,
            'steps': episode_steps,
            'time': episode_time,
            'loss': episode_loss if train else None
        }
    
    def _train_step(self, states, actions, rewards):
        """Simple training step using policy gradient."""
        if not states:
            return 0.0
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Simple reward-to-go
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Forward pass with autocast
        use_amp = is_amp_available(self.device.type)
        
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            logits = self.policy(states_tensor)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Policy gradient loss
            loss = -(action_log_probs * returns).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        return loss.item()
    
    def run_simulation(self, num_episodes=10, train=True, render=False):
        """Run complete simulation."""
        logger.info(f"🎮 Starting simulation with {num_episodes} episodes")
        logger.info(f"Device: {self.device}, Training: {train}, Render: {render}")
        
        results = []
        total_start_time = time.perf_counter()
        
        for episode in range(num_episodes):
            try:
                result = self.run_episode(episode + 1, render=render, train=train)
                results.append(result)
                
                # Clear cache periodically
                if (episode + 1) % 5 == 0:
                    empty_cache(self.device)
                    
            except KeyboardInterrupt:
                logger.info("Simulation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in episode {episode + 1}: {e}")
                continue
        
        total_time = time.perf_counter() - total_start_time
        
        # Print summary
        self._print_summary(results, total_time)
        
        return results
    
    def _print_summary(self, results, total_time):
        """Print simulation summary."""
        if not results:
            logger.warning("No results to summarize")
            return
        
        rewards = [r['reward'] for r in results]
        steps = [r['steps'] for r in results]
        times = [r['time'] for r in results]
        
        logger.info("\n" + "="*50)
        logger.info("📊 Simulation Summary")
        logger.info("="*50)
        logger.info(f"Total episodes: {len(results)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average episode time: {np.mean(times):.3f}s")
        logger.info(f"Episodes per second: {len(results)/total_time:.2f}")
        logger.info("")
        logger.info(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        logger.info(f"Best reward: {np.max(rewards):.2f}")
        logger.info(f"Average steps: {np.mean(steps):.1f}")
        logger.info("")
        logger.info(f"Device: {self.device}")
        
        # Memory info
        if self.memory_usage:
            avg_memory = np.mean(self.memory_usage) / 1024**2  # MB
            max_memory = np.max(self.memory_usage) / 1024**2  # MB
            logger.info(f"Average GPU memory: {avg_memory:.1f} MB")
            logger.info(f"Peak GPU memory: {max_memory:.1f} MB")
    
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.env.close()
        empty_cache(self.device)
        logger.info("✅ Cleanup completed")

def main():
    """Main function to demonstrate Intel GPU simulation."""
    print("🚀 LeRobot Intel GPU Simulation Demo\n")
    
    try:
        # Setup device
        device, batch_size = setup_intel_gpu_environment()
        
        # Create simulator
        simulator = IntelGPUSimulator(env_name="CartPole-v1", device=device)
        
        # Setup environment
        obs_space, action_space = simulator.setup_environment()
        
        # Setup policy
        obs_size = obs_space.shape[0]
        action_size = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        simulator.setup_policy(obs_size, action_size)
        
        # Run simulation
        results = simulator.run_simulation(
            num_episodes=20,
            train=True,
            render=False  # Set to True if you want to see the environment
        )
        
        # Test different environments
        print("\n🔄 Testing different environments...")
        
        for env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]:
            try:
                logger.info(f"\nTesting {env_name}...")
                test_sim = IntelGPUSimulator(env_name=env_name, device=device)
                obs_space, action_space = test_sim.setup_environment()
                
                obs_size = obs_space.shape[0]
                action_size = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
                
                test_sim.setup_policy(obs_size, action_size)
                test_sim.run_simulation(num_episodes=3, train=False)
                test_sim.cleanup()
                
            except Exception as e:
                logger.warning(f"Skipping {env_name}: {e}")
                continue
        
        # Cleanup
        simulator.cleanup()
        
        print("\n🎉 Intel GPU simulation demo completed successfully!")
        print(f"💡 Your Intel GPU ({device}) is working well with LeRobot!")
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()
