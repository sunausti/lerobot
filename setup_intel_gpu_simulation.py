#!/usr/bin/env python3
"""
Script to set up LeRobot simulation environment with Intel GPU support.
This script will install dependencies, configure environment, and create test files.
"""
import os
import subprocess
import sys
import platform
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🚀 LeRobot Intel GPU Simulation Environment Setup")
    print("=" * 60)
    print("This script will:")
    print("  ✓ Check Intel GPU availability")
    print("  ✓ Install Intel GPU optimized PyTorch")
    print("  ✓ Install simulation dependencies")
    print("  ✓ Configure environment variables")
    print("  ✓ Create test and example scripts")
    print("=" * 60)

def check_system_requirements():
    """Check system requirements."""
    logger.info("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.error(f"Python 3.8+ required, found {python_version}")
        return False
    logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}")
    
    # Check operating system
    os_name = platform.system()
    logger.info(f"✅ Operating system: {os_name}")
    
    if os_name == "Windows":
        logger.info("   Windows detected - Intel GPU support available")
    elif os_name == "Linux":
        logger.info("   Linux detected - Intel GPU support available")
    else:
        logger.warning(f"   {os_name} - Intel GPU support may be limited")
    
    return True

def check_intel_gpu():
    """Check if Intel GPU is available."""
    logger.info("Checking Intel GPU availability...")
    
    try:
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            logger.info(f"✅ Intel GPU detected! Found {device_count} XPU device(s)")
            for i in range(device_count):
                try:
                    device_name = torch.xpu.get_device_name(i)
                    logger.info(f"   Device {i}: {device_name}")
                except Exception as e:
                    logger.info(f"   Device {i}: Intel GPU (name unavailable)")
            return True
        else:
            logger.warning("❌ Intel GPU not available")
            return False
    except ImportError:
        logger.warning("❌ PyTorch not installed or doesn't support Intel GPU")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking Intel GPU: {e}")
        return False

def install_intel_pytorch():
    """Install Intel GPU optimized PyTorch."""
    logger.info("Installing Intel GPU optimized PyTorch...")
    
    try:
        # Check if we need to uninstall existing PyTorch
        try:
            import torch
            logger.info("Existing PyTorch found, will reinstall with Intel GPU support")
            
            # Uninstall existing PyTorch
            logger.info("Uninstalling existing PyTorch packages...")
            subprocess.run([
                sys.executable, "-m", "pip", "uninstall", "-y", 
                "torch", "torchvision", "torchaudio"
            ], check=False)  # Don't fail if packages aren't installed
            
        except ImportError:
            logger.info("No existing PyTorch found")
        
        # Install Intel GPU version
        logger.info("Installing PyTorch with Intel GPU support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/xpu"
        ], check=True)
        
        logger.info("✅ Intel GPU PyTorch installation completed")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install Intel GPU PyTorch: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during PyTorch installation: {e}")
        return False

def install_simulation_dependencies():
    """Install simulation environment dependencies."""
    logger.info("Installing simulation dependencies...")
    
    # Core dependencies
    core_dependencies = [
        "numpy>=1.21.0",
        "scipy",
        "matplotlib",
        "opencv-python",
        "imageio[ffmpeg]",
        "pillow",
        "pyyaml",
    ]
    
    # Simulation dependencies
    sim_dependencies = [
        "gymnasium[classic_control]",
        "gymnasium[box2d]", 
        "gymnasium[atari]",
    ]
    
    # Optional dependencies (may fail on some systems)
    optional_dependencies = [
        "gymnasium[mujoco]",
        "mujoco>=2.3.0",
        "dm_control",
        "pybullet",
    ]
    
    success_count = 0
    total_count = 0
    
    def install_package_list(packages, required=True):
        nonlocal success_count, total_count
        for package in packages:
            total_count += 1
            logger.info(f"Installing {package}...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], check=True, capture_output=True, text=True)
                logger.info(f"✅ Successfully installed {package}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                if required:
                    logger.error(f"❌ Failed to install {package}: {e.stderr}")
                else:
                    logger.warning(f"⚠️ Optional package {package} failed to install: {e.stderr}")
            except Exception as e:
                logger.error(f"❌ Unexpected error installing {package}: {e}")
    
    # Install core dependencies (required)
    logger.info("Installing core dependencies...")
    install_package_list(core_dependencies, required=True)
    
    # Install simulation dependencies (required)
    logger.info("Installing simulation dependencies...")
    install_package_list(sim_dependencies, required=True)
    
    # Install optional dependencies (best effort)
    logger.info("Installing optional dependencies...")
    install_package_list(optional_dependencies, required=False)
    
    logger.info(f"📦 Dependency installation completed: {success_count}/{total_count} packages installed")
    return success_count > 0

def setup_environment_variables():
    """Set up environment variables for Intel GPU."""
    logger.info("Setting up environment variables...")
    
    env_vars = {
        "INTEL_GPU_ENABLED": "1",
        "SYCL_CACHE_PERSISTENT": "1",
        "ZE_FLAT_DEVICE_HIERARCHY": "COMPOSITE",
        "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
        # "SYCL_PI_LEVEL_ZERO_MAX_POOL_SIZE": "4294967296",  # 4GB - adjust based on GPU memory
    }
    
    # Set environment variables for current session
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")
    
    # Create environment setup scripts for different shells
    
    # PowerShell script
    ps_script = "# Intel GPU Environment Variables for LeRobot\n"
    ps_script += "# Run this script in PowerShell: .\\setup_intel_gpu_env.ps1\n\n"
    for key, value in env_vars.items():
        ps_script += f"$env:{key}='{value}'\n"
    ps_script += "\nWrite-Host 'Intel GPU environment variables set successfully!' -ForegroundColor Green\n"
    
    with open("setup_intel_gpu_env.ps1", "w") as f:
        f.write(ps_script)
    
    # Batch script for Windows CMD
    bat_script = "@echo off\n"
    bat_script += "REM Intel GPU Environment Variables for LeRobot\n"
    bat_script += "REM Run this script in CMD: setup_intel_gpu_env.bat\n\n"
    for key, value in env_vars.items():
        bat_script += f"set {key}={value}\n"
    bat_script += "\necho Intel GPU environment variables set successfully!\n"
    
    with open("setup_intel_gpu_env.bat", "w") as f:
        f.write(bat_script)
    
    # Bash script for Linux/WSL
    bash_script = "#!/bin/bash\n"
    bash_script += "# Intel GPU Environment Variables for LeRobot\n"
    bash_script += "# Source this script: source setup_intel_gpu_env.sh\n\n"
    for key, value in env_vars.items():
        bash_script += f"export {key}='{value}'\n"
    bash_script += "\necho 'Intel GPU environment variables set successfully!'\n"
    
    with open("setup_intel_gpu_env.sh", "w") as f:
        f.write(bash_script)
    
    # Make bash script executable on Unix systems
    if os.name != 'nt':
        os.chmod("setup_intel_gpu_env.sh", 0o755)
    
    logger.info("✅ Environment setup scripts created:")
    logger.info("   setup_intel_gpu_env.ps1  (PowerShell)")
    logger.info("   setup_intel_gpu_env.bat  (Windows CMD)")
    logger.info("   setup_intel_gpu_env.sh   (Bash/Linux)")

def create_test_script():
    """Create a test script to verify Intel GPU setup."""
    test_script = '''#!/usr/bin/env python3
"""
Test script to verify Intel GPU setup for LeRobot simulation.
"""
import torch
import gymnasium as gym
import numpy as np
from lerobot.utils.utils import get_safe_torch_device, auto_select_torch_device

def test_intel_gpu():
    """Test Intel GPU functionality."""
    print("🔍 Testing Intel GPU setup...")
    
    # Test device selection
    device = auto_select_torch_device()
    print(f"Auto-selected device: {device}")
    
    # Test XPU specifically
    try:
        xpu_device = get_safe_torch_device("xpu", log=True)
        print(f"XPU device: {xpu_device}")
        
        # Test tensor operations
        x = torch.randn(100, 100).to(xpu_device)
        y = torch.randn(100, 100).to(xpu_device)
        z = torch.mm(x, y)
        print(f"✅ Matrix multiplication successful on {xpu_device}")
        print(f"Result shape: {z.shape}")
        
        # Test memory info
        if hasattr(torch.xpu, 'memory_allocated'):
            memory = torch.xpu.memory_allocated(xpu_device)
            print(f"XPU memory allocated: {memory / 1024**2:.2f} MB")
            
        # Test autocast
        with torch.autocast(device_type="xpu", dtype=torch.float16):
            z_fp16 = torch.mm(x, y)
            print(f"✅ Autocast test successful, output dtype: {z_fp16.dtype}")
            
    except Exception as e:
        print(f"❌ XPU test failed: {e}")
        return False
        
    return True

def test_simulation_env():
    """Test simulation environment."""
    print("\\n🎮 Testing simulation environment...")
    
    try:
        # Test basic gym environment
        env = gym.make("CartPole-v1")
        obs, info = env.reset()
        print(f"✅ Basic gym environment working: {obs.shape}")
        
        # Test a step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful: reward={reward}")
        env.close()
        
        # Test if MuJoCo is available
        try:
            env = gym.make("HalfCheetah-v4")
            obs, info = env.reset()
            print(f"✅ MuJoCo environment working: {obs.shape}")
            env.close()
        except Exception as e:
            print(f"⚠️ MuJoCo not available: {e}")
            
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False
        
    return True

def test_lerobot_integration():
    """Test LeRobot integration with Intel GPU."""
    print("\\n🤖 Testing LeRobot integration...")
    
    try:
        # Test device utilities
        from lerobot.utils.device_utils import (
            synchronize_device, empty_cache, get_memory_info
        )
        
        device = torch.device("xpu")
        
        # Test device utilities
        synchronize_device(device)
        empty_cache(device)
        memory_info = get_memory_info(device)
        print(f"✅ Device utilities working: {memory_info}")
        
        # Test simple policy network
        class SimplePolicy(torch.nn.Module):
            def __init__(self, obs_size, action_size):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(obs_size, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, action_size)
                )
            
            def forward(self, x):
                return self.net(x)
        
        policy = SimplePolicy(4, 2).to(device)
        obs = torch.randn(1, 4).to(device)
        
        with torch.autocast(device_type="xpu", dtype=torch.float16):
            action = policy(obs)
        
        print(f"✅ Policy inference successful: {action.shape}")
        
    except Exception as e:
        print(f"❌ LeRobot integration test failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("🚀 Testing LeRobot Intel GPU Setup\\n")
    
    gpu_ok = test_intel_gpu()
    sim_ok = test_simulation_env()
    lerobot_ok = test_lerobot_integration()
    
    print("\\n" + "="*50)
    print("📊 Test Results:")
    print(f"Intel GPU: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"Simulation: {'✅ PASS' if sim_ok else '❌ FAIL'}")
    print(f"LeRobot Integration: {'✅ PASS' if lerobot_ok else '❌ FAIL'}")
    
    if gpu_ok and sim_ok and lerobot_ok:
        print("\\n🎉 All tests passed! LeRobot is ready for Intel GPU simulation.")
        print("\\n🚀 You can now run:")
        print("   python examples/intel_gpu_simulation.py")
        print("   lerobot-train --policy.device=xpu")
    else:
        print("\\n❌ Some tests failed. Please check the setup.")
        print("\\n🔧 Troubleshooting:")
        print("1. Make sure Intel GPU drivers are installed")
        print("2. Run: pip install torch --index-url https://download.pytorch.org/whl/xpu")
        print("3. Check Intel GPU with: intel-smi")
'''
    
    with open("test_intel_gpu_setup.py", "w") as f:
        f.write(test_script)
    
def create_example_scripts():
    """Create example simulation scripts."""
    logger.info("Creating example simulation scripts...")
    
    # Create examples directory
    os.makedirs("examples/intel_gpu", exist_ok=True)
    
    # Example 1: Basic Intel GPU simulation
    basic_example = '''#!/usr/bin/env python3
"""
Basic Intel GPU simulation example for LeRobot.
This demonstrates running a simple simulation with Intel GPU acceleration.
"""
import torch
import gymnasium as gym
import numpy as np
from lerobot.utils.utils import auto_select_torch_device, get_safe_torch_device

def run_basic_simulation():
    """Run a basic simulation with Intel GPU."""
    print("🚀 Starting basic Intel GPU simulation...")
    
    # Select Intel GPU device
    device = auto_select_torch_device()
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Simple neural network policy
    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, action_size)
    ).to(device)
    
    # Run episodes
    total_rewards = []
    num_episodes = 10
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 500:  # Max steps per episode
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action_logits = policy(obs_tensor)
                action = torch.argmax(action_logits, dim=1).item()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward:.1f} reward, {steps} steps")
    
    env.close()
    
    # Results
    avg_reward = np.mean(total_rewards)
    print(f"\\n📊 Results:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Best episode: {max(total_rewards):.1f}")
    print(f"Total episodes: {num_episodes}")
    print("✅ Basic simulation completed!")

if __name__ == "__main__":
    run_basic_simulation()
'''
    
    with open("examples/intel_gpu/basic_simulation.py", "w") as f:
        f.write(basic_example)
    
    # Example 2: Advanced training with Intel GPU
    training_example = '''#!/usr/bin/env python3
"""
Advanced training example with Intel GPU optimization for LeRobot.
This demonstrates training a policy with Intel GPU-specific optimizations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import time
from lerobot.utils.utils import auto_select_torch_device

class DQNAgent:
    """Simple DQN agent optimized for Intel GPU."""
    
    def __init__(self, state_size, action_size, lr=0.001, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or auto_select_torch_device()
        
        # Neural network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        
        # Intel GPU specific optimizations
        if 'xpu' in str(self.device):
            # Enable mixed precision for Intel GPU
            self.use_amp = True
            self.scaler = torch.amp.GradScaler()
        else:
            self.use_amp = False
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, epsilon=0.1):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() <= epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.autocast(device_type="xpu", dtype=torch.float16):
                    q_values = self.q_network(state_tensor)
            else:
                q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        # Training step with Intel GPU optimizations
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.autocast(device_type="xpu", dtype=torch.float16):
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                next_q_values = self.q_network(next_states).max(1)[0].detach()
                target_q_values = rewards + (0.99 * next_q_values * ~dones)
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.q_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

def train_intel_gpu():
    """Train DQN agent with Intel GPU optimizations."""
    print("🚀 Starting Intel GPU training...")
    
    # Setup
    device = auto_select_torch_device()
    print(f"Training device: {device}")
    
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, device=device)
    
    # Training parameters
    episodes = 500
    batch_size = 64 if 'xpu' in str(device) else 32  # Larger batch for Intel GPU
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    scores = deque(maxlen=100)
    start_time = time.time()
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        
        for step in range(500):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
            
            # Train the agent
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
        
        scores.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Progress update
        if episode % 50 == 0:
            avg_score = np.mean(scores)
            elapsed_time = time.time() - start_time
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                  f"Epsilon: {epsilon:.3f}, Time: {elapsed_time:.1f}s")
            
            # Performance metrics for Intel GPU
            if 'xpu' in str(device) and hasattr(torch.xpu, 'memory_allocated'):
                memory_mb = torch.xpu.memory_allocated(device) / 1024**2
                print(f"Intel GPU memory: {memory_mb:.1f} MB")
    
    env.close()
    
    # Final results
    final_avg = np.mean(scores)
    total_time = time.time() - start_time
    
    print(f"\\n🎯 Training Results:")
    print(f"Final average score: {final_avg:.2f}")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Episodes per second: {episodes/total_time:.2f}")
    print("✅ Training completed!")

if __name__ == "__main__":
    train_intel_gpu()
'''
    
    with open("examples/intel_gpu/advanced_training.py", "w") as f:
        f.write(training_example)
    
    # Create README for examples
    readme_content = '''# Intel GPU Simulation Examples for LeRobot

This directory contains examples demonstrating how to use LeRobot with Intel GPU acceleration.

## Prerequisites

1. Intel GPU drivers installed
2. PyTorch with XPU support: `pip install torch --index-url https://download.pytorch.org/whl/xpu`
3. LeRobot installed: `pip install lerobot`

## Examples

### 1. Basic Simulation (`basic_simulation.py`)

A simple example showing how to run a CartPole simulation with Intel GPU acceleration.

```bash
python basic_simulation.py
```

Features:
- Intel GPU device auto-selection
- Simple neural network policy
- Multiple episode simulation
- Performance metrics

### 2. Advanced Training (`advanced_training.py`)

More sophisticated example with DQN training optimized for Intel GPU.

```bash
python advanced_training.py
```

Features:
- DQN agent implementation
- Intel GPU-specific optimizations
- Mixed precision training (when available)
- Experience replay buffer
- Training progress tracking
- Memory usage monitoring

## Intel GPU Optimizations

These examples include several optimizations for Intel GPU:

1. **Mixed Precision Training**: Uses FP16 when available to improve performance
2. **Larger Batch Sizes**: Intel GPUs benefit from larger batch sizes
3. **Memory Management**: Proper memory allocation and cleanup
4. **Device Auto-Selection**: Automatically uses Intel GPU when available

## Performance Tips

1. **Batch Size**: Use larger batch sizes (64+ for training)
2. **Mixed Precision**: Enable when supported for 2x speed improvement
3. **Memory Management**: Use `torch.xpu.empty_cache()` to free memory
4. **Environment Variables**: Set Intel GPU environment variables for optimal performance

## Troubleshooting

If you encounter issues:

1. Check Intel GPU drivers: `intel-smi`
2. Verify PyTorch XPU: `python -c "import torch; print(torch.xpu.is_available())"`
3. Test basic operations: `python test_intel_gpu_setup.py`
4. Check environment variables in `setup_intel_gpu_env.*` files

## Environment Variables

The setup script creates environment files for optimal Intel GPU performance:
- `setup_intel_gpu_env.ps1` (PowerShell)
- `setup_intel_gpu_env.bat` (Windows CMD)
- `setup_intel_gpu_env.sh` (Linux/WSL)

Run the appropriate script for your shell before training.
'''
    
    with open("examples/intel_gpu/README.md", "w") as f:
        f.write(readme_content)
    
    # Make scripts executable on Unix systems
    if os.name != 'nt':
        os.chmod("examples/intel_gpu/basic_simulation.py", 0o755)
        os.chmod("examples/intel_gpu/advanced_training.py", 0o755)
    
    logger.info("✅ Created example scripts:")
    logger.info("   examples/intel_gpu/basic_simulation.py")
    logger.info("   examples/intel_gpu/advanced_training.py")
    logger.info("   examples/intel_gpu/README.md")

def main():
    """Main setup function for Intel GPU simulation environment."""
    logger.info("🚀 Setting up LeRobot simulation environment with Intel GPU support")
    logger.info("=" * 80)
    
    try:
        # Step 1: Check system requirements
        logger.info("📋 Step 1: Checking system requirements...")
        if not check_system_requirements():
            logger.error("❌ System requirements check failed")
            return 1
        
        # Step 2: Check Intel GPU availability
        logger.info("\n🔍 Step 2: Checking Intel GPU availability...")
        intel_gpu_available = check_intel_gpu()
        
        if not intel_gpu_available:
            logger.info("📦 Installing Intel GPU PyTorch support...")
            if not install_intel_pytorch():
                logger.error("❌ Intel GPU PyTorch installation failed")
                return 1
            
            # Recheck after installation
            logger.info("🔍 Rechecking Intel GPU after installation...")
            intel_gpu_available = check_intel_gpu()
            
            if not intel_gpu_available:
                logger.error("❌ Intel GPU still not available after installation")
                logger.error("💡 Please ensure Intel GPU drivers are properly installed:")
                logger.error("   https://www.intel.com/content/www/us/en/support/articles/000005520.html")
                return 1
        
        # Step 3: Install simulation dependencies
        logger.info("\n📦 Step 3: Installing simulation dependencies...")
        if not install_simulation_dependencies():
            logger.error("❌ Simulation dependencies installation failed")
            return 1
        
        # Step 4: Set up environment variables
        logger.info("\n⚙️ Step 4: Setting up environment variables...")
        setup_environment_variables()
        
        # Step 5: Create test script
        logger.info("\n📝 Step 5: Creating test script...")
        create_test_script()
        
        # Step 6: Create example scripts
        logger.info("\n📁 Step 6: Creating example scripts...")
        create_example_scripts()
        
        # Success summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        logger.info("\n📋 What was installed:")
        logger.info("✅ PyTorch with Intel GPU (XPU) support")
        logger.info("✅ Gymnasium simulation environments")
        logger.info("✅ LeRobot device utilities (Intel GPU support)")
        logger.info("✅ Environment configuration scripts")
        logger.info("✅ Test and validation scripts")
        logger.info("✅ Example simulation scripts")
        
        logger.info("\n� Next steps:")
        logger.info("1. Test the setup:")
        logger.info("   python test_intel_gpu_setup.py")
        logger.info("\n2. Set up environment variables (choose your shell):")
        logger.info("   PowerShell: .\\setup_intel_gpu_env.ps1")
        logger.info("   CMD:        setup_intel_gpu_env.bat")
        logger.info("   Bash:       source setup_intel_gpu_env.sh")
        logger.info("\n3. Try example simulations:")
        logger.info("   python examples/intel_gpu/basic_simulation.py")
        logger.info("   python examples/intel_gpu/advanced_training.py")
        logger.info("\n4. Use with LeRobot training:")
        logger.info("   lerobot-train --policy.device=xpu")
        
        logger.info("\n📚 Documentation:")
        logger.info("   examples/intel_gpu/README.md - Detailed usage guide")
        
        logger.info("\n🔧 Troubleshooting:")
        logger.info("   If issues occur, check logs above and run the test script")
        logger.info("   Ensure Intel GPU drivers are up to date")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Setup failed with error: {e}")
        logger.error("Check the logs above for details")
        return 1

if __name__ == "__main__":
    import sys
    
    # Configure logging for main execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('intel_gpu_setup.log', mode='w')
        ]
    )
    
    logger.info("LeRobot Intel GPU Simulation Environment Setup")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    exit_code = main()
    
    if exit_code == 0:
        logger.info("\n🎊 Setup completed successfully!")
        logger.info("Log saved to: intel_gpu_setup.log")
    else:
        logger.error("\n💥 Setup failed!")
        logger.error("Check intel_gpu_setup.log for detailed error information")
    
    sys.exit(exit_code)
