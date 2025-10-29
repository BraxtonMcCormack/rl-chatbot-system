## Project Overview

This chatbot combines two fundamental machine learning approaches:

1. **Decision Tree Classification** - Classifies user intent from text input using TF-IDF vectorization
2. **Q-Learning (Reinforcement Learning)** - Learns optimal response selection through trial and error


## Technical Architecture

```
User Input → TF-IDF Vectorization → Decision Tree Classification
    ↓
Intent + Confidence → Q-Learning Agent (State + Intent)
    ↓
ε-greedy Policy → Action Selection → Reward Calculation
    ↓
Q-Table Update → State Transition → Response Generation
```

**Core Components:**

- **Intent Classifier**: Decision tree with TF-IDF features, 6 intent categories, 85-90% accuracy
- **Q-Learning Agent**: α=0.1, γ=0.9, ε=0.2, sparse Q-table with 15-30 state-action pairs
- **Dialogue Manager**: 6 states, 7 actions, intent-driven transitions

## Comparison to Modern Transformer Models


This chatbot uses old-school machine learning (decision trees and Q-learning) instead of modern methods used by commercial models. The main difference: mine is simple and you can see exactly how it works - it matches your text to one of 30 trained phrases and picks a pre-written response based on what it learned. It trains in 30 seconds on any laptop and the whole thing is tiny. My project shows how chatbots worked before transformers took over, which is useful for learning the basics but nobody would dothis anymore.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the chatbot
python train_chatbot.py 100 (bigger number won't make it smarter, low plateau)

# Interactive chat
python interactive_demo.py

# Quick test
python chatbot_system.py
```


## Customization

Hyperparameters in `chatbot_system.py`:

```python
learning_rate = 0.1        # Learning speed (try: 0.05-0.3)
discount_factor = 0.9       # Future reward weight (try: 0.7-0.99)
exploration_rate = 0.2     # Randomness (try: 0.1-0.5)
max_depth = 5             # Tree complexity (try: 3-10)
```

Add new intents in `IntentClassifier._train_classifier()`:
```python
training_data = [("phrase", "intent"), ...]
```

## Project Structure

```
├── chatbot_system.py      # Core implementation (350+ lines)
├── interactive_demo.py    # Interactive interface
├── train_chatbot.py       # Training pipeline
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Known Limitations

- Limited vocabulary (~30 training phrases)
- No semantic understanding or paraphrasing
- Context-free (no conversation history)
- Template-based responses only

## Learning Resources

- scikit-learn Decision Trees documentation

## License

MIT License - free to use for learning and experimentation.

---

**Note:** This is an educational project demonstrating traditional ML in conversational AI. For production chatbots don't do it like this.