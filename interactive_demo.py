"""
Interactive Chatbot Demo
Run this script to chat with the AI chatbot in real-time
"""

from chatbot_system import AIChatbot, DialogueState
import sys


def print_banner():
    print("\n" + "=" * 70)
    print(" " * 15 + "AI CHATBOT SYSTEM DEMO")
    print(" " * 10 + "Reinforcement Learning + Decision Trees")
    print("=" * 70)
    print("\nType 'quit' or 'exit' to end the conversation")
    print("Type 'stats' to see chatbot learning statistics")
    print("Type 'history' to see conversation history")
    print("-" * 70 + "\n")


def print_stats(chatbot):
    """Display chatbot statistics"""
    stats = chatbot.get_statistics()
    print("\n" + "=" * 50)
    print("CHATBOT LEARNING STATISTICS")
    print("=" * 50)
    print(f"Total Learning Steps: {stats['total_steps']}")
    print(f"Q-Table Size: {stats['q_table_size']} state-action pairs")
    print(f"Conversations: {stats['conversations']}")
    print(f"Current State: {DialogueState.get_state_name(chatbot.current_state)}")
    print("=" * 50 + "\n")


def print_history(chatbot):
    """Display conversation history"""
    print("\n" + "=" * 70)
    print("CONVERSATION HISTORY")
    print("=" * 70)
    
    for i, turn in enumerate(chatbot.conversation_history[-10:], 1):  # Last 10 turns
        print(f"\nTurn {i}:")
        print(f"  User: {turn['user_input']}")
        print(f"  Intent: {turn['intent']} (confidence: {turn['confidence']:.2f})")
        print(f"  State: {turn['state']} -> Action: {turn['action']}")
        print(f"  Reward: {turn['reward']:.2f}")
        print(f"  Bot: {turn['response']}")
    
    print("=" * 70 + "\n")


def main():
    """Main interactive loop"""
    print_banner()
    
    # Initialize chatbot
    chatbot = AIChatbot()
    
    # Try to load existing model
    chatbot.load_model()
    
    print("Chatbot: Hello! I'm ready to chat. How can I help you today?\n")
    
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nChatbot: Thank you for chatting! Saving my learning progress...")
                chatbot.save_model()
                print("Goodbye!\n")
                break
            
            if user_input.lower() == 'stats':
                print_stats(chatbot)
                continue
            
            if user_input.lower() == 'history':
                print_history(chatbot)
                continue
            
            # Generate response with training enabled
            response, intent, confidence, action = chatbot.generate_response(
                user_input, 
                training_mode=True
            )
            
            # Display response
            print(f"Chatbot: {response}")
            
            # Show debug info (optional - comment out for cleaner output)
            print(f"[Debug: Intent={intent}, Confidence={confidence:.2f}, Action={action}]\n")
            
            conversation_count += 1
            
            # Periodically save progress
            if conversation_count % 10 == 0:
                chatbot.save_model()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving progress...")
            chatbot.save_model()
            print("Goodbye!\n")
            break
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing...\n")


if __name__ == "__main__":
    main()
