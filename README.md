# 🧾 AI Expense Scanner

A comprehensive Flask-based web application that uses AI and OCR to automatically scan, categorize, track expenses, and provide intelligent price comparisons from receipt photos.

## ✨ Key Features

### 📸 **Smart Receipt Processing**
- **Drag & Drop Upload** - Easy photo upload with CSRF protection
- **Multi-Method OCR** - OCR.space API + Tesseract fallback + demo mode
- **Intelligent Text Parsing** - Advanced receipt format recognition
- **Auto Image Resizing** - Handles large images (auto-resize for API limits)

### 🤖 **AI-Powered Analysis** 
- **Expense Categorization** - Rule-based + Hugging Face AI models
- **Merchant Detection** - Direct OCR extraction without pattern matching
- **Address Extraction** - Parse store locations from receipts
- **Grocery Item Recognition** - Extract individual items with prices

### 💰 **Smart Price Comparison**
- **Location-Aware Comparisons** - Find better prices at nearby stores
- **Historical Price Tracking** - Track price trends over time
- **Intelligent Suggestions** - Personalized shopping recommendations
- **Bulk Buying Analysis** - Identify savings opportunities

### 📍 **Location Features**
- **Address Detection** - Extract store addresses from receipts
- **Nearby Store Finding** - Locate similar stores in your area
- **Distance Calculations** - Haversine formula for accurate distances
- **Multi-Region Support** - US and Canadian address formats

### 📊 **Analytics & Insights**
- **Expense Statistics** - Category breakdowns and spending analysis
- **Price Trends** - Historical price data and volatility analysis
- **Merchant Insights** - Compare pricing across different stores
- **Savings Opportunities** - Identify where you can save money

### 🔒 **Security & Performance**
- **CSRF Protection** - Secure file uploads and form handling
- **Input Sanitization** - Safe handling of OCR text and user input
- **Environment Variables** - Secure API key management
- **SQLite Database** - Local data storage with optimized queries

## 🏗️ Architecture

### **Database Schema**
- **expenses** - Main transaction records with OCR text
- **user_profile** - User preferences and location data
- **local_stores** - Store locations with coordinates and details
- **store_inventory** - Product availability and pricing per store
- **grocery_items** - Individual items extracted from receipts
- **price_comparisons** - Historical price comparison results
- **price_history** - Price tracking for trend analysis

### **OCR Processing Pipeline**
1. **Image Upload & Validation** - File type and size checks
2. **Auto-Resize** - Compress large images for API compatibility
3. **Multi-Method OCR** - Try OCR.space → Tesseract → Demo fallback
4. **Text Processing** - Clean and normalize OCR results

### **Price Comparison Engine**
1. **Grocery Extraction** - Parse items with proximity-based price matching
2. **Database Matching** - Compare against local inventory and price history
3. **Location Analysis** - Factor in distance and store accessibility
4. **Savings Calculation** - Generate actionable recommendations

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd ai-expense-scanner
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Configure API Keys (REQUIRED)

Create a `.env` file in the project root:
```bash
# Required API Keys
OCR_SPACE_API_KEY=your_ocr_space_key_here
HUGGINGFACE_API_TOKEN=hf_your_huggingface_token_here
FLASK_SECRET_KEY=your-super-secret-random-key-change-in-production

# Optional Configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

**🔑 Get Your API Keys (Both FREE):**

1. **OCR.space API Key**:
   - Visit: https://ocr.space/ocrapi
   - Sign up for free (25,000 requests/month)
   - Copy your API key

2. **Hugging Face Token**:
   - Visit: https://huggingface.co/settings/tokens
   - Create account and generate token
   - Copy the token (starts with `hf_`)

### 3. Run the Application
```bash
source venv/bin/activate
python expense_scanner_python.py
```

### 4. Access the Web Interface
Open your browser and visit: **http://localhost:5000**

## 📋 Dependencies

The application requires these Python packages (see `requirements.txt`):

### Core Dependencies
- **Flask 3.1.2** - Web framework
- **Flask-WTF 1.2.2** - CSRF protection and form handling
- **python-dotenv 1.1.1** - Environment variable management

### Image & OCR Processing
- **Pillow 11.3.0** - Image processing and resizing
- **requests 2.32.5** - HTTP requests for OCR APIs
- **pytesseract** - Local OCR processing (optional)

### Built-in Python Modules
- `sqlite3` - Database operations
- `base64` - Image encoding for APIs
- `json` - Data serialization
- `datetime` - Timestamp handling
- `re` - Pattern matching and text processing
- `os` - File and environment operations
- `mimetypes` - File type detection

## 🛠️ Advanced Configuration

### OCR Setup Options

**Option 1: OCR.space API (Recommended)**
```bash
# In .env file
OCR_SPACE_API_KEY=your_actual_api_key_here
```

**Option 2: Local Tesseract OCR**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Python package
pip install pytesseract
```

**Option 3: Demo Mode**
- Uses hardcoded sample text for development
- Automatically enabled when no valid API key is configured

### Database Initialization
The application automatically creates and initializes the SQLite database on first run with:
- Sample store locations (Toronto, San Jose, Los Angeles)
- Sample inventory data for price comparisons
- All required tables and indexes

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OCR_SPACE_API_KEY` | **Yes** | `helloworld` | OCR.space API key for text extraction |
| `HUGGINGFACE_API_TOKEN` | No | - | Hugging Face token for AI categorization |
| `FLASK_SECRET_KEY` | **Yes** | Auto-generated | Flask session security |
| `FLASK_ENV` | No | `development` | Flask environment mode |
| `FLASK_DEBUG` | No | `True` | Enable debug mode |

## 📊 Usage Examples

### Basic Receipt Processing
1. **Upload Receipt** - Drag/drop or click to upload receipt image
2. **Auto-Processing** - OCR extracts text, categorizes expense
3. **View Results** - See amount, merchant, category, and nearby stores

### Advanced Features
- **Price Comparisons** - Automatically compare grocery prices with local stores
- **Location Services** - Find nearby stores based on receipt address
- **Spending Analysis** - View statistics and category breakdowns
- **Historical Tracking** - Monitor price trends over time

## 🔧 Development

### Project Structure
```
ai-expense-scanner/
├── expense_scanner_python.py    # Main Flask application (1800+ lines)
├── templates/
│   └── index.html              # Complete web interface with price features
├── uploads/                    # Temporary file uploads
├── expenses.db                # SQLite database (auto-created)
├── .env                       # Your API keys (create from example)
├── requirements.txt           # Python dependencies
├── OCR_SETUP.md              # Detailed OCR configuration guide
└── test files/               # Development testing scripts
```

### Key Functions
- **`extract_text_from_image()`** - Multi-method OCR processing
- **`categorize_expense_free()`** - Rule-based expense categorization
- **`extract_grocery_items()`** - Advanced item extraction with price matching
- **`get_price_comparisons()`** - Location-aware price comparison engine
- **`find_stores_by_address()`** - Address-based store location
- **`extract_store_address()`** - Parse addresses from receipt text

### Adding New Features
The codebase is well-documented with comprehensive function comments. Key extension points:
- **New Categories** - Edit categorization rules in `categorize_expense_free()`
- **Additional OCR Methods** - Add to `extract_text_from_image()`
- **New Store Chains** - Add to `populate_sample_stores()`
- **Price Sources** - Extend `get_price_comparisons()` with new data sources

## 🔒 Security Features

- ✅ **CSRF Protection** - All forms protected against cross-site attacks
- ✅ **File Upload Security** - Size limits, type validation, safe storage
- ✅ **SQL Injection Prevention** - Parameterized queries throughout
- ✅ **Input Sanitization** - OCR text and user input properly cleaned
- ✅ **API Key Security** - Environment variables, never hardcoded
- ✅ **Error Handling** - Graceful degradation and informative messages

## 🐛 Troubleshooting

### Common Issues

**OCR Not Working**
- Check your OCR.space API key in `.env`
- Verify image is under 16MB and valid format
- Check console for detailed error messages
- Fallback to Tesseract or demo mode available

**Price Comparisons Empty**
- Ensure grocery items are being extracted (check console output)
- Verify sample inventory is populated (automatic on first run)
- Current implementation works best with common grocery items

**Location Features Not Working**
- Address extraction works best with clear store headers
- Sample stores are available for Toronto, San Jose, Los Angeles
- Distance calculations require latitude/longitude data

**Database Issues**
- Delete `expenses.db` to reset (will be recreated automatically)
- Check write permissions in project directory
- SQLite requires no additional setup

### Debug Mode
Enable detailed logging by setting:
```bash
# In .env
FLASK_ENV=development
FLASK_DEBUG=True
```

### Performance Tips
- Large images are automatically resized for API compatibility
- Database queries are optimized with appropriate indexes
- OCR caching reduces redundant API calls
- Sample data provides immediate functionality

## 📈 API Limits & Costs

| Service | Free Tier | Paid Options |
|---------|-----------|-------------|
| **OCR.space** | 25,000 requests/month | $29.90/month for 100K |
| **Hugging Face** | 1,000 requests/month | $9/month for 10K |
| **Tesseract** | Unlimited (local) | System resources only |

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Configure environment** (copy `.env.example` to `.env` with your keys)
4. **Test thoroughly** - Use provided test scripts
5. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
6. **Push and create Pull Request**

### Development Tools Included
- `test_core_functions.py` - Test OCR and categorization
- `debug_price_comparison.py` - Debug price matching logic
- `analyze_receipt_format.py` - Analyze OCR text patterns

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## ⚠️ Important Setup Notes

- 🔑 **API keys are required** - Get free keys from OCR.space and Hugging Face
- 📁 **Create .env file** - Copy configuration from environment variables section
- 🔒 **Never commit .env** - Contains your personal API keys
- 🏪 **Sample data included** - Works immediately with demo stores and inventory
- 🧪 **Test scripts provided** - Use for debugging and development

---

**Need Help?** Check the troubleshooting section above or open an issue.