#!/bin/bash
#for transperency I used chat gpt to make this setup script since I never think to make them look pretty

# AI Chatbot Setup Script
echo "================================================"
echo "  AI Chatbot System - Setup & Installation"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Error: Python 3 is not installed."
    echo "Please install Python 3.7 or higher."
    exit 1
fi

echo "✓ Python found"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt --break-system-packages

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies."
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Run a quick test
echo "Running quick functionality test..."
python3 chatbot_system.py > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Error: Test failed."
    exit 1
fi

echo "✓ System test passed"
echo ""

echo "================================================"
echo "  ✓ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Train the chatbot (optional, already pre-trained):"
echo "   python3 train_chatbot.py 100"
echo ""
echo "2. Start interactive chat:"
echo "   python3 interactive_demo.py"
echo ""
echo "3. Run quick demo:"
echo "   python3 chatbot_system.py"
echo ""
echo "================================================"
