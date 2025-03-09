import os
import yaml
import argparse
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from src.environment.blackjack_env import BlackjackEnv
from src.agents.dqn_agent import DQNAgent

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_env_and_agent(config):
    """Create environment and agent instances."""
    # Create environment
    env = BlackjackEnv(
        num_players=config['environment']['num_players'],
        num_decks=config['environment']['num_decks'],
        initial_bankroll=config['environment']['initial_bankroll']
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, config)
    
    return env, agent

def train(config_path, resume=False, checkpoint_path=None):
    """Main training loop."""
    # Load configuration
    config = load_config(config_path)
    
    # Create environment and agent
    env, agent = create_env_and_agent(config)
    
    # Load checkpoint if resuming
    if resume and checkpoint_path:
        agent.load(checkpoint_path)
        print(f"Resumed training from {checkpoint_path}")
    
    # Setup logging
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(config['logging']['tensorboard_dir'], current_time)
    writer = SummaryWriter(log_dir)
    
    # Training parameters
    num_episodes = config['agent']['training']['num_episodes']
    log_interval = config['logging']['log_interval']
    save_model_path = config['logging']['save_model_path']
    
    # Ensure save directory exists
    os.makedirs(save_model_path, exist_ok=True)
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            done = False
            
            # Reset noise for noisy networks
            if config['agent']['network']['use_noisy_nets']:
                agent.policy_net.reset_noise()
                agent.target_net.reset_noise()
            
            while not done:
                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Update agent
                loss = agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    env.bankroll,
                    env.initial_bankroll
                )
                
                state = next_state
                episode_reward += reward
                episode_loss += loss
                
                # Check if we should end episode early
                if not env.should_continue_session()[0]:
                    done = True
            
            # Update metrics
            agent.update_metrics(episode_reward)
            metrics = agent.get_metrics()
            
            # Log metrics
            if episode % log_interval == 0:
                print(f"Episode {episode}")
                print(f"Running reward: {metrics['running_reward']:.2f}")
                print(f"Episode reward: {episode_reward:.2f}")
                print(f"Average loss: {episode_loss:.6f}")
                print(f"Learning rate: {metrics['learning_rate']:.6f}")
                print(f"Bankroll: {env.bankroll:.2f}")
                print(f"Win rate: {env.wins/(env.wins + env.losses):.2%}")
                print("-" * 50)
                
                # Log to tensorboard
                writer.add_scalar('Reward/running', metrics['running_reward'], episode)
                writer.add_scalar('Reward/episode', episode_reward, episode)
                writer.add_scalar('Loss/average', episode_loss, episode)
                writer.add_scalar('Metrics/learning_rate', metrics['learning_rate'], episode)
                writer.add_scalar('Metrics/bankroll', env.bankroll, episode)
                writer.add_scalar('Metrics/win_rate', env.wins/(env.wins + env.losses), episode)
                
                # Log risk metrics
                for metric, value in metrics['risk_metrics'].items():
                    writer.add_scalar(f'Risk/{metric}', value, episode)
            
            # Save best model
            if metrics['running_reward'] > metrics['best_reward']:
                save_path = os.path.join(save_model_path, 'best_model.pth')
                agent.save(save_path)
                print(f"Saved best model with reward {metrics['running_reward']:.2f}")
            
            # Save checkpoint periodically
            if episode % 1000 == 0:
                save_path = os.path.join(save_model_path, f'checkpoint_{episode}.pth')
                agent.save(save_path)
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Save final model
        save_path = os.path.join(save_model_path, 'final_model.pth')
        agent.save(save_path)
        print("Saved final model")
        
        # Close tensorboard writer
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Blackjack DQN agent')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint file for resuming training')
    
    args = parser.parse_args()
    train(args.config, args.resume, args.checkpoint) 