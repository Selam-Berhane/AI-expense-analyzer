#!/bin/bash
# =============================================================================
# AI Expense Scanner - Setup and Installation Script
# =============================================================================
# 
# This script sets up the AI Expense Scanner project with all dependencies
# and required configuration files.
#
# Usage: bash requirements.sh
# =============================================================================

set -e  # Exit on any error

echo "🚀 Setting up AI Expense Scanner..."
echo ""

# Check if we're in the right directory
if [[ ! -f "expense_scanner_python.py" ]]; then
    echo "❌ Please run this script from the project root directory"
    echo "   (The directory containing expense_scanner_python.py)"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p templates uploads
echo "   ✅ Created templates/ and uploads/ directories"

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate
echo "   ✅ Virtual environment activated"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements from requirements.txt
if [[ -f "requirements.txt" ]]; then
    echo "📦 Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "   ✅ All dependencies installed"
else
    echo "📦 Installing Python dependencies..."
    pip install Flask==3.1.2 Pillow==11.3.0 requests==2.32.5 Flask-WTF==1.2.2 python-dotenv==1.1.1
    echo "   ✅ All dependencies installed"
fi

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo ""
    echo "🔑 Setting up environment variables..."
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        echo "   ✅ Copied .env.example to .env"
        echo ""
        echo "⚠️  IMPORTANT: You must edit your .env file with your API keys!"
        echo "   📝 Edit .env and add your Hugging Face token"
        echo "   🔗 Get free token at: https://huggingface.co/settings/tokens"
    else
        echo "   ❌ .env.example not found - please create .env manually"
    fi
else
    echo "   ✅ .env file already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🔑 REQUIRED: Configure your API keys"
echo "============================================="
echo "1. Edit your .env file:"
echo "   - Add your Hugging Face API token (FREE)"
echo "   - Optionally add OCR.space API key (FREE)"
echo ""
echo "2. Get your FREE Hugging Face token:"
echo "   📱 Visit: https://huggingface.co/settings/tokens"
echo "   🔑 Create new token with 'Read' access"
echo "   📝 Copy token to HUGGINGFACE_API_TOKEN in .env"
echo ""
echo "📋 Next steps:"
echo "============================================="
echo "1. Edit .env file with your API keys"
echo "2. Run: python expense_scanner_python.py"
echo "3. Open: http://localhost:5000"
echo "4. Upload receipt images and enjoy! 📸"
echo ""
echo "🎯 Features included:"
echo "============================================="
echo "✅ Drag & drop receipt upload"
echo "✅ OCR text extraction (OCR.space API)"
echo "✅ AI-powered expense categorization (Hugging Face)"
echo "✅ Local SQLite database storage"
echo "✅ Real-time expense statistics & charts"
echo "✅ Recent expenses history"
echo "✅ CSRF protection & security features"
echo "✅ Responsive web design (mobile-friendly)"
echo ""
echo "📚 Need help? Check README.md for detailed instructions!"
echo ""

# Check if environment validation will pass
echo "🔍 Checking environment configuration..."
if source venv/bin/activate && python -c "
from dotenv import load_dotenv
import os
load_dotenv()

# Check required variables
required_vars = ['HUGGINGFACE_API_TOKEN', 'FLASK_SECRET_KEY']
missing = []

for var in required_vars:
    value = os.getenv(var)
    if not value or 'your_token_here' in value or 'your-' in value or 'change' in value:
        missing.append(var)

if missing:
    print(f'⚠️  Missing configuration: {', '.join(missing)}')
    print('   Please edit your .env file before running the app')
    exit(1)
else:
    print('✅ Environment configuration looks good!')
"; then
    echo ""
    echo "🚀 Ready to start! Run: python expense_scanner_python.py"
else
    echo ""
    echo "⚠️  Please configure your .env file before starting the app"
    echo "   See the instructions above ☝️"
fi

echo ""
echo "============================================="
echo "🎉 AI Expense Scanner setup complete!"
echo "============================================="