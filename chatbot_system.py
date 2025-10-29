"""
AI Chatbot System with Reinforcement Learning and Decision Trees
A conversational AI that learns optimal dialogue strategies using Q-learning
and classifies user intents with decision trees.
"""

import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import random


class DialogueState:
    """Represents the current state of the conversation"""
    GREETING = 0
    ASKING_PRODUCT = 1
    ASKING_PRICE = 2
    ASKING_SUPPORT = 3
    PROVIDING_INFO = 4
    CLOSING = 5
    
    @staticmethod
    def get_state_name(state):
        names = {0: "GREETING", 1: "ASKING_PRODUCT", 2: "ASKING_PRICE", 
                3: "ASKING_SUPPORT", 4: "PROVIDING_INFO", 5: "CLOSING"}
        return names.get(state, "UNKNOWN")


class IntentClassifier:
    """Decision Tree based intent classifier"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=50)
        self.classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.intents = ['greeting', 'product_inquiry', 'price_inquiry', 
                       'support_request', 'thank', 'goodbye']
        self._train_classifier()
    
    def _train_classifier(self):
        """Train the decision tree with sample data"""
        training_data = [
            # Greetings
            ("hello", "greeting"), ("hi there", "greeting"), ("hey", "greeting"),
            ("good morning", "greeting"), ("good afternoon", "greeting"),
            
            # Product inquiries
            ("what products do you have", "product_inquiry"),
            ("tell me about your products", "product_inquiry"),
            ("what do you sell", "product_inquiry"),
            ("show me your items", "product_inquiry"),
            ("what can i buy", "product_inquiry"),
            
            # Price inquiries
            ("how much does it cost", "price_inquiry"),
            ("what is the price", "price_inquiry"),
            ("how expensive", "price_inquiry"),
            ("pricing information", "price_inquiry"),
            ("cost of product", "price_inquiry"),
            
            # Support requests
            ("i need help", "support_request"),
            ("can you help me", "support_request"),
            ("having problems", "support_request"),
            ("support needed", "support_request"),
            ("assistance required", "support_request"),
            
            # Thanks
            ("thank you", "thank"), ("thanks", "thank"),
            ("appreciate it", "thank"), ("that helps", "thank"),
            
            # Goodbye
            ("bye", "goodbye"), ("goodbye", "goodbye"),
            ("see you later", "goodbye"), ("have a nice day", "goodbye"),
        ]
        
        texts, labels = zip(*training_data)
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
    
    def classify(self, user_input):
        """Classify user intent using decision tree"""
        X = self.vectorizer.transform([user_input.lower()])
        intent = self.classifier.predict(X)[0]
        confidence = max(self.classifier.predict_proba(X)[0])
        return intent, confidence


class ReinforcementLearningAgent:
    """Q-Learning agent for dialogue policy optimization"""
    

# ========== HYPERPARAMETER DETAIL COMMENT FOR MESSING WITH HERE ==========
# learning_rate (α): How fast the bot learns (try: 0.05-0.3)
# discount_factor (γ): How much it values future rewards (try: 0.7-0.99)
# exploration_rate (ε): How often it tries random actions (try: 0.1-0.5)
# =======================================================    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Q-table: Q(state, action) values
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Define possible actions (dialogue responses)
        self.actions = [
            'greet_user',
            'ask_what_they_need',
            'provide_product_info',
            'provide_price_info',
            'offer_support',
            'ask_if_helpful',
            'say_goodbye'
        ]
        
        # Track learning statistics
        self.episode_rewards = []
        self.total_steps = 0
    
    def get_action(self, state, intent):
        """Select action using epsilon-greedy policy"""
        state_key = (state, intent)
        
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Exploitation: best known action
        q_values = self.q_table[state_key]
        if not q_values:
            return random.choice(self.actions)
        
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state, intent, action, reward, next_state, next_intent):
        """Update Q-value using Q-learning formula"""
        current_key = (state, intent)
        next_key = (next_state, next_intent)
        
        # Current Q-value
        current_q = self.q_table[current_key][action]
        
        # Maximum Q-value for next state
        next_q_values = self.q_table[next_key]
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[current_key][action] = new_q
        
        self.total_steps += 1
    
    def get_reward(self, state, action, intent):
        """Calculate reward for state-action pair"""
        # Reward shaping for good dialogue flow
        rewards = {
            (DialogueState.GREETING, 'greet_user'): 1.0,
            (DialogueState.GREETING, 'ask_what_they_need'): 0.8,
            (DialogueState.ASKING_PRODUCT, 'provide_product_info'): 1.0,
            (DialogueState.ASKING_PRICE, 'provide_price_info'): 1.0,
            (DialogueState.ASKING_SUPPORT, 'offer_support'): 1.0,
            (DialogueState.PROVIDING_INFO, 'ask_if_helpful'): 0.8,
            (DialogueState.CLOSING, 'say_goodbye'): 1.0,
        }
        
        # Intent-based rewards
        if intent == 'goodbye' and action == 'say_goodbye':
            return 1.0
        if intent == 'thank' and action == 'ask_if_helpful':
            return 0.9
        
        return rewards.get((state, action), -0.1)


class AIChatbot:
    """Main chatbot system integrating all components"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.rl_agent = ReinforcementLearningAgent()
        self.current_state = DialogueState.GREETING
        self.conversation_history = []
        
        # Response templates
        self.responses = {
            'greet_user': [
                "Hello! Welcome to our AI Chatbot System. How can I assist you today?",
                "Hi there! I'm here to help. What can I do for you?",
                "Greetings! How may I help you today?"
            ],
            'ask_what_they_need': [
                "What would you like to know about?",
                "How can I assist you further?",
                "What information are you looking for?"
            ],
            'provide_product_info': [
                "We offer a range of AI-powered solutions including chatbots, recommendation systems, and analytics tools.",
                "Our main products include conversational AI, machine learning models, and data analysis platforms.",
            ],
            'provide_price_info': [
                "Our pricing starts at $99/month for basic plans, with enterprise solutions available. Would you like detailed pricing?",
                "We have flexible pricing from $99-$999/month depending on your needs. I can provide more details!",
            ],
            'offer_support': [
                "I'm here to help! Can you tell me more about the issue you're experiencing?",
                "I'd be happy to assist. What specific support do you need?",
            ],
            'ask_if_helpful': [
                "Was this information helpful? Is there anything else you'd like to know?",
                "I hope that helps! Do you have any other questions?",
            ],
            'say_goodbye': [
                "Thank you for chatting! Have a great day!",
                "Goodbye! Feel free to reach out anytime!",
            ]
        }
    
    def get_next_state(self, intent, action):
        """Determine next dialogue state based on intent and action"""
        state_transitions = {
            'greeting': DialogueState.ASKING_PRODUCT,
            'product_inquiry': DialogueState.PROVIDING_INFO,
            'price_inquiry': DialogueState.PROVIDING_INFO,
            'support_request': DialogueState.ASKING_SUPPORT,
            'thank': DialogueState.CLOSING,
            'goodbye': DialogueState.CLOSING,
        }
        
        if action == 'say_goodbye':
            return DialogueState.CLOSING
        
        return state_transitions.get(intent, self.current_state)
    
    def generate_response(self, user_input, training_mode=False):
        """Generate response using RL agent and intent classifier"""
        # Classify user intent
        intent, confidence = self.intent_classifier.classify(user_input)
        
        # Get action from RL agent
        action = self.rl_agent.get_action(self.current_state, intent)
        
        # Calculate reward
        reward = self.rl_agent.get_reward(self.current_state, action, intent)
        
        # Determine next state
        next_state = self.get_next_state(intent, action)
        
        # Update Q-values if in training mode
        if training_mode:
            self.rl_agent.update_q_value(
                self.current_state, intent, action, 
                reward, next_state, intent
            )
        
        # Generate response
        response = random.choice(self.responses[action])
        
        # Update state
        prev_state = self.current_state
        self.current_state = next_state
        
        # Log conversation
        self.conversation_history.append({
            'user_input': user_input,
            'intent': intent,
            'confidence': confidence,
            'state': DialogueState.get_state_name(prev_state),
            'action': action,
            'reward': reward,
            'response': response
        })
        
        return response, intent, confidence, action
    
    def save_model(self, filepath='chatbot_model.pkl'):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.rl_agent.q_table),
                'total_steps': self.rl_agent.total_steps,
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='chatbot_model.pkl'):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.rl_agent.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
                self.rl_agent.total_steps = data['total_steps']
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print("No saved model found. Starting fresh.")
    
    def get_statistics(self):
        """Get chatbot learning statistics"""
        return {
            'total_steps': self.rl_agent.total_steps,
            'q_table_size': len(self.rl_agent.q_table),
            'conversations': len(self.conversation_history)
        }


if __name__ == "__main__":
    # Example usage
    chatbot = AIChatbot()
    
    print("=" * 60)
    print("AI Chatbot System - Proof of Concept")
    print("Using Reinforcement Learning + Decision Trees")
    print("=" * 60)
    print()
    
    # Demonstrate the chatbot
    test_inputs = [
        "Hello!",
        "What products do you offer?",
        "How much does it cost?",
        "Thank you!",
        "Goodbye"
    ]
    
    for user_input in test_inputs:
        print(f"User: {user_input}")
        response, intent, confidence, action = chatbot.generate_response(user_input, training_mode=True)
        print(f"Bot: {response}")
        print(f"[Intent: {intent} (confidence: {confidence:.2f}), Action: {action}]")
        print()
    
    # Show statistics
    stats = chatbot.get_statistics()
    print(f"\nChatbot Statistics:")
    print(f"- Total learning steps: {stats['total_steps']}")
    print(f"- Q-table entries: {stats['q_table_size']}")
    print(f"- Conversations handled: {stats['conversations']}")
