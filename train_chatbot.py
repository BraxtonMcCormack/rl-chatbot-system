"""
Chatbot Training Script
Pre-train the RL agent with simulated conversations
"""

from chatbot_system import AIChatbot
import random
import matplotlib.pyplot as plt


def generate_training_conversations():
    """Generate diverse training conversation scenarios"""
    scenarios = [
        # Scenario 1: Product inquiry
        [
            "Hi there",
            "What products do you have?",
            "Interesting, what about pricing?",
            "Thanks for the info",
            "Goodbye"
        ],
        
        # Scenario 2: Price-focused
        [
            "Hello",
            "How much does it cost?",
            "What do you sell?",
            "Thank you",
            "Bye"
        ],
        
        # Scenario 3: Support request
        [
            "Hey",
            "I need some help",
            "What products can you help with?",
            "That helps, thanks",
            "See you later"
        ],
        
        # Scenario 4: Quick inquiry
        [
            "Hi",
            "Tell me about your products",
            "Thanks",
            "Goodbye"
        ],
        
        # Scenario 5: Detailed inquiry
        [
            "Good morning",
            "What do you offer?",
            "What's the pricing?",
            "Can you help me with setup?",
            "That's helpful",
            "Have a nice day"
        ],
        
        # Scenario 6: Price then product
        [
            "Hello there",
            "How expensive are your products?",
            "What exactly do you sell?",
            "Appreciate it",
            "Goodbye"
        ],
    ]
    
    return scenarios


def train_chatbot(num_episodes=100, verbose=True):
    """Train the chatbot using simulated conversations"""
    chatbot = AIChatbot()
    training_scenarios = generate_training_conversations()
    
    episode_rewards = []
    
    print(f"Starting training for {num_episodes} episodes...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        # Select random scenario
        scenario = random.choice(training_scenarios)
        
        # Reset chatbot state
        chatbot.current_state = 0  # Reset to GREETING
        
        episode_reward = 0
        
        if verbose and episode % 10 == 0:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print("-" * 40)
        
        # Run through conversation
        for turn, user_input in enumerate(scenario, 1):
            response, intent, confidence, action = chatbot.generate_response(
                user_input,
                training_mode=True
            )
            
            # Get reward from last conversation turn
            if chatbot.conversation_history:
                reward = chatbot.conversation_history[-1]['reward']
                episode_reward += reward
                
                if verbose and episode % 10 == 0:
                    print(f"  Turn {turn}: {user_input[:30]}")
                    print(f"    Intent: {intent}, Action: {action}, Reward: {reward:.2f}")
        
        episode_rewards.append(episode_reward)
        
        if verbose and episode % 10 == 0:
            print(f"  Total Episode Reward: {episode_reward:.2f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Average reward over last 10 episodes: {sum(episode_rewards[-10:]) / 10:.2f}")
    
    # Save trained model
    chatbot.save_model()
    
    # Display statistics
    stats = chatbot.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"- Total learning steps: {stats['total_steps']}")
    print(f"- Q-table size: {stats['q_table_size']} state-action pairs")
    print(f"- Total conversations: {stats['conversations']}")
    
    return chatbot, episode_rewards


def plot_training_progress(episode_rewards):
    """Visualize training progress"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
        
        # Moving average
        window = 10
        if len(episode_rewards) >= window:
            moving_avg = [sum(episode_rewards[max(0, i-window):i+1]) / min(i+1, window) 
                         for i in range(len(episode_rewards))]
            plt.plot(moving_avg, linewidth=2, label=f'{window}-Episode Moving Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('RL Agent Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/training_progress.png', dpi=150, bbox_inches='tight')
        print(f"\nTraining plot saved to outputs/training_progress.png")
    except Exception as e:
        print(f"Could not create plot: {e}")


def evaluate_chatbot(chatbot, test_scenarios=None):
    """Evaluate trained chatbot performance"""
    print("\n" + "=" * 60)
    print("EVALUATING CHATBOT PERFORMANCE")
    print("=" * 60)
    
    if test_scenarios is None:
        test_scenarios = [
            ["Hi", "What do you sell?", "Pricing?", "Thanks", "Bye"],
            ["Hello", "I need help", "Tell me about products", "Thank you", "Goodbye"],
        ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nTest Scenario {i}:")
        print("-" * 40)
        
        chatbot.current_state = 0  # Reset state
        total_reward = 0
        
        for user_input in scenario:
            response, intent, confidence, action = chatbot.generate_response(
                user_input,
                training_mode=False  # No learning during evaluation
            )
            
            reward = chatbot.conversation_history[-1]['reward']
            total_reward += reward
            
            print(f"User: {user_input}")
            print(f"Bot: {response}")
            print(f"  [{intent}, {action}, reward: {reward:.2f}]")
        
        print(f"\nScenario Total Reward: {total_reward:.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_episodes = 100
    if len(sys.argv) > 1:
        try:
            num_episodes = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of episodes. Using default: {num_episodes}")
    
    # Train the chatbot
    chatbot, rewards = train_chatbot(num_episodes=num_episodes, verbose=True)
    
    # Plot training progress
    plot_training_progress(rewards)
    
    # Evaluate performance
    evaluate_chatbot(chatbot)
    
    print("\n✓ Training complete! Run 'python interactive_demo.py' to chat with the trained bot.")
