# app.py - AI Expense Scanner Flask App
from flask import Flask, render_template, request, jsonify
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv
import os
import base64
import json
from datetime import datetime
import sqlite3
import re
from PIL import Image
import io
import requests
import mimetypes

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key-change-me')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CSRF protection
csrf = CSRFProtect(app)

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SQLite database
def init_db():
    """
    Initialize the SQLite database with all required tables for the expense scanner.
    
    Creates tables for:
    - expenses: Main transaction records
    - user_profile: User preferences and location data
    - local_stores: Store location and contact information
    - store_inventory: Product availability and pricing per store
    - grocery_items: Individual items extracted from grocery receipts
    - price_comparisons: Historical price comparison results
    - price_history: Price tracking for trend analysis
    """
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    
    # Expenses table
    c.execute('''CREATE TABLE IF NOT EXISTS expenses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  amount REAL,
                  category TEXT,
                  merchant TEXT,
                  date TEXT,
                  raw_text TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # User profile table (enhanced with location data)
    c.execute('''CREATE TABLE IF NOT EXISTS user_profile
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  diet_type TEXT,
                  allergies TEXT,
                  cooking_skill TEXT,
                  preferred_cuisines TEXT,
                  latitude REAL,
                  longitude REAL,
                  address TEXT,
                  zip_code TEXT,
                  city TEXT,
                  state TEXT,
                  location_updated_at TIMESTAMP,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Local stores table for location-aware price comparisons
    c.execute('''CREATE TABLE IF NOT EXISTS local_stores
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  store_name TEXT NOT NULL,
                  chain_name TEXT,
                  address TEXT,
                  city TEXT,
                  state TEXT,
                  zip_code TEXT,
                  latitude REAL,
                  longitude REAL,
                  phone TEXT,
                  store_hours TEXT,
                  store_type TEXT,
                  distance_miles REAL,
                  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Store inventory table to track which items are available at which stores
    c.execute('''CREATE TABLE IF NOT EXISTS store_inventory
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  store_id INTEGER,
                  item_name TEXT,
                  category TEXT,
                  brand TEXT,
                  size TEXT,
                  unit_type TEXT,
                  price REAL,
                  price_per_unit REAL,
                  stock_quantity INTEGER DEFAULT 0,
                  in_stock BOOLEAN DEFAULT 1,
                  last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(store_id) REFERENCES local_stores(id))''')
    
    # Grocery items table
    c.execute('''CREATE TABLE IF NOT EXISTS grocery_items
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  expense_id INTEGER,
                  item_name TEXT,
                  quantity TEXT,
                  category TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(expense_id) REFERENCES expenses(id))''')
    
    
    # Price tracking table
    c.execute('''CREATE TABLE IF NOT EXISTS price_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  grocery_item_id INTEGER,
                  merchant_name TEXT,
                  item_name TEXT,
                  price REAL,
                  quantity TEXT,
                  unit_type TEXT,
                  price_per_unit REAL,
                  location TEXT,
                  date TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(grocery_item_id) REFERENCES grocery_items(id))''')
    
    # Price comparisons table
    c.execute('''CREATE TABLE IF NOT EXISTS price_comparisons
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  expense_id INTEGER,
                  item_name TEXT,
                  current_price REAL,
                  current_merchant TEXT,
                  best_price REAL,
                  best_merchant TEXT,
                  savings_amount REAL,
                  savings_percentage REAL,
                  suggestion TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(expense_id) REFERENCES expenses(id))''')
    
    # Merchant locations table
    c.execute('''CREATE TABLE IF NOT EXISTS merchant_locations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  merchant_name TEXT,
                  location TEXT,
                  address TEXT,
                  distance_miles REAL,
                  avg_price_index REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

def extract_store_address(lines):
    """
    Extract store address information from receipt text lines.
    
    Parses receipt text to identify and extract address components including:
    - Street address
    - City and state/province
    - Postal/ZIP codes (Canadian and US formats)
    - Phone numbers
    
    Args:
        lines (list): List of text lines from OCR-processed receipt
        
    Returns:
        dict: Address information with keys for street_address, city, state, 
              postal_code, phone, and full_address
    """
    import re
    
    address_info = {
        'street_address': None,
        'city': None,
        'state': None,
        'postal_code': None,
        'phone': None,
        'full_address': None
    }
    
    # Common address patterns
    phone_pattern = r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
    postal_code_pattern = r'([A-Z]\d[A-Z][\s-]?\d[A-Z]\d)'  # Canadian postal code
    zip_code_pattern = r'(\d{5}(?:-\d{4})?)'  # US zip code
    
    # Look through all lines for address components
    potential_address_lines = []
    
    for i, line in enumerate(lines):
        line_cleaned = line.strip()
        line_lower = line_cleaned.lower()
        
        # Skip obvious non-address lines
        if any(skip in line_lower for skip in ['total', 'subtotal', 'tax', 'thank you', 'customer copy', '***']):
            continue
            
        # Look for phone numbers
        phone_match = re.search(phone_pattern, line_cleaned)
        if phone_match:
            address_info['phone'] = phone_match.group(1)
            potential_address_lines.append((i, line_cleaned, 'phone'))
            continue
        
        # Look for postal/zip codes
        postal_match = re.search(postal_code_pattern, line_cleaned, re.IGNORECASE)
        zip_match = re.search(zip_code_pattern, line_cleaned)
        if postal_match:
            address_info['postal_code'] = postal_match.group(1).upper()
            potential_address_lines.append((i, line_cleaned, 'postal'))
        elif zip_match:
            address_info['postal_code'] = zip_match.group(1)
            potential_address_lines.append((i, line_cleaned, 'zip'))
            
        # Look for street addresses (contains numbers and street indicators)
        street_indicators = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'blvd', 'blud', 'boulevard', 'drive', 'dr', 'lane', 'ln', 'lakeshore', 'king', 'queen', 'front']
        if (re.search(r'\d+', line_cleaned) and 
            any(indicator in line_lower for indicator in street_indicators) and
            len(line_cleaned.split()) >= 2 and
            'hst' not in line_lower and 'tax' not in line_lower):  # Avoid tax lines
            
            # Clean up common OCR errors in street addresses  
            cleaned_address = line_cleaned
            cleaned_address = cleaned_address.replace(' BLUD ', ' BLVD ')  # Fix OCR error
            cleaned_address = cleaned_address.replace(' VEST ', ' WEST ')  # Fix OCR error
            # Remove phone numbers from address
            cleaned_address = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', cleaned_address).strip()
            
            if cleaned_address:
                address_info['street_address'] = cleaned_address
                potential_address_lines.append((i, line_cleaned, 'street'))
            
        # Special handling for Toronto store name lines
        if 'toronto' in line_lower:
            address_info['city'] = 'Toronto'
            address_info['state'] = 'ON'
    
    # Try to extract city and state/province from lines near postal codes
    for i, line, line_type in potential_address_lines:
        if line_type in ['postal', 'zip']:
            # Look at the same line and nearby lines for city/state
            for check_i in range(max(0, i-2), min(len(lines), i+3)):
                check_line = lines[check_i].strip()
                
                # Canadian provinces
                provinces = ['ON', 'BC', 'AB', 'SK', 'MB', 'QC', 'NB', 'NS', 'PE', 'NL', 'YT', 'NT', 'NU']
                # US states (abbreviated)
                states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
                
                # Check if line contains province/state
                words = check_line.upper().split()
                for word in words:
                    if word in provinces or word in states:
                        address_info['state'] = word
                        # The word before state is likely the city
                        word_index = words.index(word)
                        if word_index > 0:
                            address_info['city'] = words[word_index - 1].title()
                        break
    
    # Build full address string
    address_parts = []
    if address_info['street_address']:
        address_parts.append(address_info['street_address'])
    if address_info['city'] and address_info['state']:
        address_parts.append(f"{address_info['city']}, {address_info['state']}")
    if address_info['postal_code']:
        address_parts.append(address_info['postal_code'])
    
    if address_parts:
        address_info['full_address'] = ', '.join(address_parts)
    
    print(f"📍 Extracted address info: {address_info}")
    return address_info

def extract_merchant_name(lines):
    """
    Extract merchant name directly from OCR text without pattern matching.
    
    Searches the first few lines of receipt text to identify the store name,
    filtering out common receipt headers and non-merchant text.
    
    Args:
        lines (list): List of text lines from OCR-processed receipt
        
    Returns:
        str: Merchant name in uppercase format, or "Unknown" if not found
    """
    if not lines:
        return "Unknown"
    
    # Look through the first few lines to find the store name
    for i, line in enumerate(lines[:3]):  # Check first 3 lines
        line_lower = line.lower()
        
        # Skip lines that are clearly not store names
        skip_words = ['date:', 'time:', 'receipt #', 'store #', 'address', 'phone', 'tel:', 'www.']
        if any(skip_word in line_lower for skip_word in skip_words):
            continue
        
        # Skip lines that are just numbers or short codes
        if len(line.strip()) < 3 or line.strip().isdigit():
            continue
            
        # Clean up the merchant name but keep it as OCR read it
        cleaned = line.strip()
        
        # Remove common prefixes/suffixes that add noise but keep the core name
        # Only remove obvious noise, not the actual store name
        cleaned = cleaned.replace('***', '').replace('---', '')  # Remove decorative characters
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        
        # If it looks like a reasonable merchant name, use it
        if 2 <= len(cleaned) <= 50 and not cleaned.isdigit():
            return cleaned.upper()
    
    # Fallback: use first non-empty line
    first_line = lines[0] if lines else "Unknown"
    return first_line.strip().upper() if first_line.strip() else "Unknown"

def extract_text_with_tesseract(image_path):
    """
    Extract text using local Tesseract OCR (no API key needed).
    
    Alternative OCR method that runs locally using Tesseract engine.
    Requires tesseract and pytesseract to be installed on the system.
    
    Args:
        image_path (str): Path to the image file to process
        
    Returns:
        str: Extracted text from image, or None if extraction fails
    """
    try:
        import pytesseract
        from PIL import Image
        
        # Open and process image
        image = Image.open(image_path)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(image)
        print(f"✅ Tesseract OCR Success - Extracted {len(text)} characters")
        return text.strip()
        
    except ImportError:
        print("⚠️ Tesseract not installed. Install with: pip install pytesseract")
        return None
    except Exception as e:
        print(f"❌ Tesseract OCR Error: {e}")
        return None

# OCR with multiple fallback options
def resize_image_for_ocr(image_path, max_size_kb=800):
    """
    Resize image if it's too large for OCR API processing.
    
    Many OCR APIs have file size limits (typically 1MB). This function
    automatically resizes images that exceed the specified limit while
    maintaining quality and aspect ratio.
    
    Args:
        image_path (str): Path to the original image file
        max_size_kb (int): Maximum allowed file size in kilobytes (default: 800KB)
        
    Returns:
        str: Path to the resized image file, or original path if no resize needed
    """
    try:
        from PIL import Image
        import io
        
        # Check current file size
        file_size_kb = os.path.getsize(image_path) / 1024
        print(f"📏 Image size: {file_size_kb:.1f} KB")
        
        if file_size_kb <= max_size_kb:
            print("✅ Image size is within limits")
            return image_path
        
        print(f"🔄 Resizing image from {file_size_kb:.1f} KB to fit under {max_size_kb} KB...")
        
        # Open and resize image
        with Image.open(image_path) as img:
            # Calculate resize ratio to get under size limit
            ratio = (max_size_kb / file_size_kb) ** 0.5  # Square root for 2D scaling
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            
            print(f"📐 Resizing from {img.width}x{img.height} to {new_width}x{new_height}")
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save resized version
            resized_path = image_path.replace('.jpg', '_resized.jpg').replace('.png', '_resized.png')
            resized_img.save(resized_path, optimize=True, quality=85)
            
            new_size_kb = os.path.getsize(resized_path) / 1024
            print(f"✅ Resized to {new_size_kb:.1f} KB")
            
            return resized_path
            
    except ImportError:
        print("⚠️ PIL not available for resizing. Install with: pip install pillow")
        return image_path
    except Exception as e:
        print(f"❌ Error resizing image: {e}")
        return image_path

def extract_text_from_image(image_path):
    """
    Extract text from image using multiple OCR methods with fallback options.
    
    Tries multiple OCR approaches in order of preference:
    1. OCR.space API (if valid API key is configured)
    2. Local Tesseract OCR (if installed)
    3. Fallback to demo text (for development)
    
    Args:
        image_path (str): Path to the image file to process
        
    Returns:
        str: Extracted text from the image, or fallback demo text if OCR fails
    """
    
    # Method 1: Try OCR.space API if we have a real key
    ocr_key = os.getenv('OCR_SPACE_API_KEY', 'helloworld')
    if ocr_key != 'helloworld' and len(ocr_key) > 10:  # Real API key
        try:
            # Resize image if needed
            processed_image = resize_image_for_ocr(image_path)
            
            # Convert image to base64
            with open(processed_image, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            
            # OCR.space API
            url = 'https://api.ocr.space/parse/image'
            payload = {
                'apikey': ocr_key,
                'base64Image': f'data:image/jpeg;base64,{img_data}',
                'language': 'eng',
                'isOverlayRequired': False,
                'OCREngine': 2
            }
            
            response = requests.post(url, data=payload, timeout=30)
            print(f"🌐 OCR API Response Status: {response.status_code}")
            
            result = response.json()
            print(f"🔍 OCR API Result Keys: {list(result.keys())}")
            
            if result.get('ParsedResults'):
                print(f"📄 ParsedResults found: {len(result['ParsedResults'])} results")
                if result['ParsedResults'][0].get('ParsedText'):
                    extracted_text = result['ParsedResults'][0]['ParsedText'].strip()
                    print(f"✅ OCR.space API Success - {len(extracted_text)} characters")
                    print(f"🔍 OCR Text Preview: {extracted_text[:200]}...")
                    return extracted_text
                else:
                    print("❌ No ParsedText in result")
            else:
                print("❌ No ParsedResults in response")
                print(f"📝 Full response: {result}")
                
        except Exception as e:
            print(f"⚠️ OCR.space API failed: {e}")
            print(f"📝 API Response: {result if 'result' in locals() else 'No response received'}")
    
    # Method 2: Try local Tesseract OCR
    print("🔄 Trying local Tesseract OCR...")
    tesseract_result = extract_text_with_tesseract(image_path)
    if tesseract_result and len(tesseract_result) > 20:  # Got decent text
        return tesseract_result
    
    # Method 3: Final fallback - should rarely be used with real API key
    print("⚠️ All OCR methods failed. Check your setup:")
    print("   - OCR.space API key validity")
    print("   - Internet connection")
    print("   - Image quality and format")
    
    # Simple fallback for development
    return "SAMPLE STORE\nDATE: 12/15/2024\nITEM $10.00\nTOTAL: $10.00"

# Free AI categorization using Ollama (runs locally) or Hugging Face
def categorize_expense_free(raw_text):
    """
    Categorize expense using free rule-based pattern matching and text analysis.
    
    Analyzes receipt text to extract:
    - Transaction amount (using multiple regex patterns)
    - Merchant name (from receipt header)
    - Expense category (based on merchant patterns and keywords)
    - Store address information (for location-based features)
    
    Args:
        raw_text (str): Raw OCR text from the receipt
        
    Returns:
        dict: Expense data with amount, category, merchant, address_info, and confidence
    """
    
    # Extract amount using improved regex patterns
    amount_patterns = [
        r'TOTAL[:\s]*\$?(\d+\.?\d{2})',       # TOTAL: $12.34 or TOTAL $12.34
        r'AMOUNT DUE[:\s]*\$?(\d+\.?\d{2})',  # AMOUNT DUE: $12.34
        r'BALANCE[:\s]*\$?(\d+\.?\d{2})',     # BALANCE: $12.34  
        r'GRAND TOTAL[:\s]*\$?(\d+\.?\d{2})', # GRAND TOTAL: $12.34
        r'\$(\d+\.\d{2})\s*$',                # $12.34 at end of line
        r'(\d+\.\d{2})\s*$'                   # 12.34 at end of line
    ]
    
    amount = 0.0
    print(f"💰 Looking for total amount in text...")
    
    for i, pattern in enumerate(amount_patterns):
        matches = re.findall(pattern, raw_text, re.IGNORECASE | re.MULTILINE)
        print(f"   Pattern {i+1}: {pattern} → Found: {matches}")
        if matches:
            try:
                # Take the last/largest amount found
                potential_amount = float(matches[-1])
                if potential_amount > amount:  # Keep the largest reasonable amount
                    amount = potential_amount
                    print(f"   ✅ Using amount: ${amount}")
            except ValueError as e:
                print(f"   ❌ Could not convert '{matches[-1]}' to float: {e}")
                continue
    
    print(f"💰 Final extracted amount: ${amount}")
    
    # Extract merchant and address with improved logic
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    print(f"🏪 Extracting merchant from {len(lines)} lines:")
    for i, line in enumerate(lines[:5]):
        print(f"   Line {i+1}: '{line}'")
    merchant = extract_merchant_name(lines)
    print(f"🏪 Extracted merchant: '{merchant}'")
    
    # Extract store address from receipt
    address_info = extract_store_address(lines)
    
    # Simple category classification based on keywords
    text_lower = raw_text.lower()
    
    categories = {
        'Groceries': ['walmart', 'target', 'grocery', 'supermarket', 'kroger', 'safeway', 'whole foods', 'costco', 'ralphs', 'vons', 'trader joe'],
        'Food & Dining': ['restaurant', 'cafe', 'pizza', 'burger', 'dining', 'mcdonalds', 'subway', 'starbucks', 'chipotle'],
        'Gas & Fuel': ['shell', 'exxon', 'chevron', 'bp', 'gas', 'fuel', 'station', 'mobil'],
        'Shopping': ['amazon', 'ebay', 'mall', 'retail', 'shopping', 'purchase'],
        'Healthcare': ['pharmacy', 'cvs', 'walgreens', 'hospital', 'clinic', 'medical', 'doctor'],
        'Transportation': ['uber', 'lyft', 'taxi', 'metro', 'bus', 'train', 'parking'],
        'Entertainment': ['movie', 'cinema', 'netflix', 'spotify', 'game', 'entertainment', 'theater'],
        'Utilities': ['electric', 'gas bill', 'water', 'internet', 'phone', 'utility'],
        'Other': []
    }
    
    # Flexible category detection based on merchant name and receipt content
    merchant_lower = merchant.lower()
    text_lower = raw_text.lower()
    
    # Check if merchant name contains grocery store indicators
    grocery_indicators = ['walmart', 'target', 'safeway', 'whole foods', 'kroger', 'costco', 
                         'ralphs', 'vons', 'trader joe', 'grocery', 'market', 'food', 'supermarket']
    
    category = 'Other'  # Default category
    
    # First check merchant name for category clues
    if any(indicator in merchant_lower for indicator in grocery_indicators):
        category = 'Groceries'
    else:
        # Then check entire receipt text for category keywords
        for cat, keywords in categories.items():
            # Check both merchant name and full text
            if any(keyword in merchant_lower for keyword in keywords) or \
               any(keyword in text_lower for keyword in keywords):
                category = cat
                break
    
    return {
        'amount': amount,
        'category': category,
        'merchant': merchant,
        'address_info': address_info,
        'confidence': 0.85 if category != 'Other' else 0.60
    }

# Alternative: Use Hugging Face free inference API
def categorize_with_huggingface(raw_text):
    """
    Use Hugging Face free inference API for expense categorization.
    
    Alternative categorization method using BART model for zero-shot classification.
    Requires HUGGINGFACE_API_TOKEN environment variable.
    
    Args:
        raw_text (str): Raw OCR text from the receipt
        
    Returns:
        tuple: (category_name, confidence_score)
    """
    try:
        # Using free Hugging Face inference API
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        hf_token = os.getenv('HUGGINGFACE_API_TOKEN')
        if not hf_token:
            print("Warning: HUGGINGFACE_API_TOKEN not set in environment variables")
            return 'Other', 0.5
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        candidate_labels = ["food", "groceries", "gas", "shopping", "transportation", "healthcare", "entertainment", "utilities"]
        
        payload = {
            "inputs": raw_text,
            "parameters": {"candidate_labels": candidate_labels}
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            best_category = result['labels'][0]
            confidence = result['scores'][0]
            
            # Map to our category names
            category_map = {
                'food': 'Food & Dining',
                'groceries': 'Groceries', 
                'gas': 'Gas & Fuel',
                'shopping': 'Shopping',
                'transportation': 'Transportation',
                'healthcare': 'Healthcare',
                'entertainment': 'Entertainment',
                'utilities': 'Utilities'
            }
            
            return category_map.get(best_category, 'Other'), confidence
        
    except Exception as e:
        print(f"HuggingFace API Error: {e}")
    
    # Fallback to rule-based
    return 'Other', 0.5

# Extract grocery items from receipt text
def extract_grocery_items(raw_text):
    """
    Enhanced grocery item extraction for various receipt formats.
    
    Parses receipt text to identify individual grocery items with their prices.
    Uses sophisticated matching to handle receipts where items and prices
    appear on separate lines (common in many store formats).
    
    Process:
    1. Find all grocery item lines using keyword matching
    2. Find all price lines using regex patterns
    3. Match items to prices using proximity-based logic
    4. Extract quantity, unit type, and calculate price per unit
    
    Args:
        raw_text (str): Raw OCR text from the receipt
        
    Returns:
        list: List of dictionaries containing item details (name, quantity, 
              unit_type, price, price_per_unit, raw_line)
    """
    import re
    
    lines = raw_text.split('\n')
    grocery_items = []
    
    # Common grocery item patterns - expanded for better matching
    grocery_keywords = [
        # Produce
        'apple', 'banana', 'orange', 'tomato', 'lettuce', 'carrot', 'onion', 'potato',
        'broccoli', 'spinach', 'cucumber', 'pepper', 'avocado', 'lemon', 'lime',
        'cilantro', 'rosemary', 'jalep', 'jalapeno',  # From actual receipt
        
        # Proteins  
        'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'eggs', 'milk', 'cheese',
        'yogurt', 'tofu', 'beans', 'lentils', 'nuts', 'almonds', 'halal', 'grnd', 'thghs',
        'feta', 'brie',  # From actual receipt
        
        # Grains & Carbs
        'bread', 'rice', 'pasta', 'cereal', 'oats', 'quinoa', 'flour', 'tortilla',
        'fusili', 'buckwheat', 'baking soda',  # From actual receipt
        
        # Pantry items
        'oil', 'olive oil', 'vinegar', 'salt', 'pepper', 'sugar', 'honey', 'garlic',
        'onion powder', 'paprika', 'cumin', 'oregano', 'basil', 'thyme',
        
        # Canned/Packaged
        'tomato sauce', 'coconut milk', 'broth', 'stock', 'canned tomatoes',
        'sanpellegrino'  # From actual receipt
    ]
    
    # Step 1: Find all grocery item lines (with their line numbers)
    item_lines = []
    for i, line in enumerate(lines):
        line_clean = line.strip()
        line_lower = line_clean.lower()
        
        # Skip empty lines, headers, payment info
        if not line_clean or any(skip in line_lower for skip in [
            'total', 'subtotal', 'cash', 'card', 'visa', 'credit', 'auth', 'ref',
            'account', 'purchase', 'datetime', 'customer copy', 'retain', 'points',
            'balance', 'approved', 'thank you', 'statement', 'validation'
        ]):
            continue
            
        # Look for grocery items
        for keyword in grocery_keywords:
            if keyword in line_lower:
                item_lines.append({
                    'line_num': i,
                    'text': line_clean,
                    'keyword': keyword,
                    'line_lower': line_lower
                })
                break
    
    # Step 2: Find all price lines
    price_lines = []
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Look for various price patterns
        price_patterns = [
            r'^\$?(\d+\.\d{2})$',                    # Just a price like "6.23"
            r'^\$(\d+\.\d{2})$',                     # Just a dollar price like "$3.49"
            r'^\d+\s+\$(\d+\.\d{2})$',               # "3 0 $1.29" format
            r'^(\d+\.\d{2})\s*$',                    # Price at end "1.29"
            r'\$(\d+\.\d{2})\s+ea',                  # "$3.49 ea or" format
            r'\$(\d+\.\d{2})\s+\w+',                 # "$3.49 ea" format
            r'(\d+)\s+\d+\s+\$(\d+\.\d{2})',        # "3 0 $1.29" format (capture price)
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, line_clean)
            if match:
                # Handle different capture group patterns
                if len(match.groups()) == 2:  # Pattern with 2 groups like "3 0 $1.29"
                    price_value = float(match.group(2))
                else:  # Pattern with 1 group
                    price_value = float(match.group(1))
                
                # Filter out unreasonably high prices (likely not individual items)
                if 0.10 <= price_value <= 50.00:  # Reasonable range for grocery items
                    price_lines.append({
                        'line_num': i,
                        'price': price_value,
                        'text': line_clean
                    })
                break
    
    print(f"🛒 Found {len(item_lines)} item lines and {len(price_lines)} price lines")
    
    # Step 3: Match items to prices using proximity and context
    for item_info in item_lines:
        item_line_num = item_info['line_num'] 
        item_text = item_info['text']
        keyword = item_info['keyword']
        
        # Look for price on same line first
        price_on_same_line = re.search(r'\$?(\d+\.\d{2})', item_text)
        if price_on_same_line:
            price = float(price_on_same_line.group(1))
        else:
            # Look for price in nearby lines (within 10 lines after the item)
            price = None
            for price_info in price_lines:
                price_line_num = price_info['line_num']
                # Price should be after item line and within reasonable distance
                if item_line_num < price_line_num <= item_line_num + 15:
                    price = price_info['price']
                    print(f"  📝 Matched '{item_text}' (line {item_line_num}) → ${price} (line {price_line_num})")
                    break
            
            if price is None:
                price = 0.0  # Default if no price found
        
        # Extract quantity and unit
        quantity_match = re.search(r'(\d+(?:\.\d+)?)\s*(lb|lbs|oz|kg|g|ct|each|ea|pk|pack)', item_info['line_lower'])
        if quantity_match:
            quantity_num = float(quantity_match.group(1))
            unit_type = quantity_match.group(2)
        else:
            # Check for "2 PK" format in the item name
            pk_match = re.search(r'(\d+)\s*pk', item_info['line_lower'])
            if pk_match:
                quantity_num = float(pk_match.group(1))
                unit_type = "pack"
            else:
                quantity_num = 1.0
                unit_type = "each"
        
        # Calculate price per unit
        price_per_unit = price / quantity_num if quantity_num > 0 else price
        
        grocery_items.append({
            'name': keyword.title(),
            'quantity': quantity_num,
            'unit_type': unit_type,
            'price': price,
            'price_per_unit': price_per_unit,
            'raw_line': item_text
        })
    
    print(f"✅ Extracted {len(grocery_items)} grocery items with prices")
    return grocery_items


# Helper functions for user profile management
def get_user_profile():
    """Get user profile including location data"""
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        c.execute("""SELECT diet_type, allergies, cooking_skill, preferred_cuisines, 
                           latitude, longitude, address, zip_code, city, state, location_updated_at 
                    FROM user_profile ORDER BY created_at DESC LIMIT 1""")
        result = c.fetchone()
        conn.close()
        
        if result:
            return {
                'diet_type': result[0],
                'allergies': result[1],
                'cooking_skill': result[2],
                'preferred_cuisines': result[3],
                'latitude': result[4],
                'longitude': result[5],
                'address': result[6],
                'zip_code': result[7],
                'city': result[8],
                'state': result[9],
                'location_updated_at': result[10]
            }
        return None
    except Exception as e:
        print(f"Error getting user profile: {e}")
        return None

def save_grocery_items(expense_id, grocery_items):
    """Save grocery items to database"""
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        for item in grocery_items:
            c.execute("""INSERT INTO grocery_items (expense_id, item_name, quantity, category)
                        VALUES (?, ?, ?, ?)""",
                     (expense_id, item['name'], str(item['quantity']), 'grocery'))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving grocery items: {e}")


# Price comparison and analysis functions
def save_price_history(grocery_items, merchant_name, location="Unknown"):
    """
    Save price data for historical tracking and trend analysis.
    
    Records grocery item prices by merchant for historical price tracking,
    enabling price comparison features and trend analysis.
    
    Args:
        grocery_items (list): List of grocery item dictionaries with prices
        merchant_name (str): Name of the merchant/store
        location (str): Store location (default: "Unknown")
    """
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        for item in grocery_items:
            # Get the grocery_item_id if it exists
            c.execute("SELECT id FROM grocery_items WHERE item_name = ? ORDER BY created_at DESC LIMIT 1", 
                     (item['name'],))
            result = c.fetchone()
            grocery_item_id = result[0] if result else None
            
            c.execute("""INSERT INTO price_history 
                        (grocery_item_id, merchant_name, item_name, price, quantity, unit_type, 
                         price_per_unit, location, date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (grocery_item_id, merchant_name, item['name'], item['price'],
                      str(item['quantity']), item['unit_type'], item['price_per_unit'],
                      location, datetime.now().strftime('%Y-%m-%d')))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving price history: {e}")

def get_price_comparisons(grocery_items, current_merchant, user_location=None):
    """
    Enhanced location-aware price comparison with local store data.
    
    Compares current grocery prices against:
    1. Local store inventory (if user location available)
    2. Historical price data from other merchants
    3. Market averages and trends
    
    Generates detailed savings suggestions with reliability scores and
    merchant recommendations based on actual price differences.
    
    Args:
        grocery_items (list): List of grocery items with current prices
        current_merchant (str): Name of the merchant where items were purchased
        user_location (tuple): Optional (latitude, longitude) for local comparisons
        
    Returns:
        list: List of price comparison dictionaries with savings information
    """
    comparisons = []
    
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        # Get user location if available
        user_lat, user_lon = None, None
        if user_location:
            user_lat, user_lon = user_location
        else:
            # Try to get from user profile
            c.execute('SELECT latitude, longitude FROM user_profile ORDER BY id DESC LIMIT 1')
            location_result = c.fetchone()
            if location_result and location_result[0] and location_result[1]:
                user_lat, user_lon = location_result[0], location_result[1]
        
        for item in grocery_items:
            item_name = item['name']
            current_price_per_unit = item['price_per_unit']
            
            # First priority: Check local store inventory if location is available
            local_comparisons = []
            if user_lat and user_lon:
                # Find nearby stores with this item in stock
                c.execute('''SELECT ls.store_name, ls.chain_name, ls.address, ls.city, 
                                   ls.latitude, ls.longitude, ls.distance_miles,
                                   si.price_per_unit, si.stock_quantity, si.last_updated
                            FROM local_stores ls
                            JOIN store_inventory si ON ls.id = si.store_id
                            WHERE LOWER(si.item_name) LIKE LOWER(?) 
                            AND si.stock_quantity > 0
                            AND ls.latitude IS NOT NULL AND ls.longitude IS NOT NULL''', 
                         (f'%{item_name}%',))
                
                nearby_inventory = c.fetchall()
                
                for store_data in nearby_inventory:
                    store_name, chain, address, city, store_lat, store_lon, _, price, stock, updated = store_data
                    
                    # Calculate distance
                    distance = calculate_distance(user_lat, user_lon, store_lat, store_lon)
                    
                    if distance <= 15:  # Within 15 miles
                        if price < current_price_per_unit and current_price_per_unit > 0:
                            savings = (current_price_per_unit - price) * item['quantity']
                            savings_pct = ((current_price_per_unit - price) / current_price_per_unit) * 100
                            
                            local_comparisons.append({
                                'item_name': item_name,
                                'current_price': item['price'],
                                'current_price_per_unit': current_price_per_unit,
                                'current_merchant': current_merchant,
                                'best_price_per_unit': price,
                                'best_merchant': store_name,
                                'chain_name': chain,
                                'store_address': f"{address}, {city}",
                                'distance_miles': distance,
                                'savings_amount': round(savings, 2),
                                'savings_percentage': round(savings_pct, 1),
                                'stock_quantity': stock,
                                'suggestion': f"Save ${savings:.2f} ({savings_pct:.1f}%) at {store_name}",
                                'suggestion_type': 'local_store',
                                'quantity': item['quantity'],
                                'unit_type': item['unit_type'],
                                'reliability_score': 95,  # High reliability for local inventory
                                'data_source': 'local_inventory'
                            })
            
            # Second priority: Historical price data for broader comparison
            historical_comparisons = []
            c.execute("""SELECT merchant_name, 
                               AVG(price_per_unit) as avg_price, 
                               MIN(price_per_unit) as min_price,
                               MAX(price_per_unit) as max_price,
                               COUNT(*) as price_count,
                               MAX(date) as last_seen
                        FROM price_history 
                        WHERE item_name = ? AND merchant_name != ?
                        GROUP BY merchant_name
                        HAVING price_count >= 1
                        ORDER BY avg_price ASC""",
                     (item_name, current_merchant))
            
            other_prices = c.fetchall()
            
            if other_prices:
                market_avg = sum(row[1] for row in other_prices) / len(other_prices)
                
                for merchant_data in other_prices[:3]:  # Top 3 alternatives
                    merchant_name, avg_price, min_price, max_price, count, last_seen = merchant_data
                    
                    if avg_price < current_price_per_unit:
                        savings_per_unit = current_price_per_unit - avg_price
                        total_savings = savings_per_unit * item['quantity']
                        savings_percentage = (savings_per_unit / current_price_per_unit) * 100 if current_price_per_unit > 0 else 0
                        
                        # Calculate price reliability (lower variance = more reliable)
                        price_variance = max_price - min_price if max_price > min_price else 0
                        reliability_score = max(0, 100 - (price_variance / avg_price * 100)) if avg_price > 0 else 50
                        
                        # Enhanced suggestion logic
                        if savings_percentage > 25:
                            suggestion_type = "excellent"
                            suggestion = f"🚨 Excellent savings! Save ${total_savings:.2f} ({savings_percentage:.1f}%) at {merchant_name}"
                        elif savings_percentage > 15:
                            suggestion_type = "significant"
                            suggestion = f"💡 Great deal at {merchant_name}! Save ${total_savings:.2f} ({savings_percentage:.1f}%)"
                        elif savings_percentage > 8:
                            suggestion_type = "moderate"
                            suggestion = f"💰 Good savings at {merchant_name}: ${total_savings:.2f} ({savings_percentage:.1f}%)"
                        else:
                            suggestion_type = "minor"
                            suggestion = f"ℹ️ Small savings at {merchant_name}: ${total_savings:.2f}"
                        
                        # Add market position context
                        if avg_price < market_avg * 0.9:
                            suggestion += " (Below market average!)"
                        
                        historical_comparisons.append({
                            'item_name': item_name,
                            'current_price': item['price'],
                            'current_price_per_unit': current_price_per_unit,
                            'current_merchant': current_merchant,
                            'best_price_per_unit': avg_price,
                            'best_merchant': merchant_name,
                            'savings_amount': round(total_savings, 2),
                            'savings_percentage': round(savings_percentage, 1),
                            'suggestion': suggestion,
                            'suggestion_type': suggestion_type,
                            'quantity': item['quantity'],
                            'unit_type': item['unit_type'],
                            'reliability_score': round(reliability_score, 1),
                            'market_position': 'below_average' if avg_price < market_avg * 0.9 else 'average',
                            'data_points': count,
                            'last_seen': last_seen,
                            'data_source': 'historical'
                        })
            
            # Combine and prioritize local vs historical data
            all_item_comparisons = []
            
            # Local store data gets priority
            if local_comparisons:
                all_item_comparisons.extend(local_comparisons[:2])  # Top 2 local deals
            
            # Add historical data if we don't have enough local data
            if len(all_item_comparisons) < 2 and historical_comparisons:
                needed = 2 - len(all_item_comparisons)
                all_item_comparisons.extend(historical_comparisons[:needed])
            
            # Add all comparisons for this item
            comparisons.extend(all_item_comparisons)
        
        conn.close()
        return comparisons
        
    except Exception as e:
        print(f"Error getting price comparisons: {e}")
        return []

def save_price_comparisons(expense_id, comparisons):
    """Save price comparison results to database"""
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        for comp in comparisons:
            c.execute("""INSERT INTO price_comparisons 
                        (expense_id, item_name, current_price, current_merchant, 
                         best_price, best_merchant, savings_amount, savings_percentage, suggestion)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (expense_id, comp['item_name'], comp['current_price'],
                      comp['current_merchant'], comp['best_price_per_unit'],
                      comp['best_merchant'], comp['savings_amount'],
                      comp['savings_percentage'], comp['suggestion']))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving price comparisons: {e}")

def get_merchant_price_trends(merchant_name, days=30):
    """
    Get price trends for a specific merchant over time.
    
    Analyzes historical price data to identify trends, volatility,
    and pricing patterns for a specific merchant.
    
    Args:
        merchant_name (str): Name of the merchant to analyze
        days (int): Number of days to look back (default: 30)
        
    Returns:
        list: Price trend data with volatility and change metrics
    """
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        c.execute("""SELECT item_name, AVG(price_per_unit) as avg_price, 
                           COUNT(*) as price_count,
                           MIN(price_per_unit) as lowest_price,
                           MAX(price_per_unit) as highest_price
                    FROM price_history 
                    WHERE merchant_name = ? 
                    AND date >= date('now', '-{} days')
                    GROUP BY item_name
                    HAVING price_count >= 2
                    ORDER BY item_name""".format(days),
                 (merchant_name,))
        
        trends = []
        for row in c.fetchall():
            price_volatility = ((row[4] - row[3]) / row[3]) * 100 if row[3] > 0 else 0
            trends.append({
                'item_name': row[0],
                'avg_price': row[1],
                'price_count': row[2],
                'lowest_price': row[3],
                'highest_price': row[4],
                'price_volatility': price_volatility
            })
        
        conn.close()
        return trends
        
    except Exception as e:
        print(f"Error getting merchant trends: {e}")
        return []


def generate_shopping_suggestions(comparisons):
    """
    Generate enhanced shopping suggestions with actionable insights.
    
    Analyzes price comparison data to provide actionable shopping advice
    including:
    - Best alternative stores for specific items
    - Bulk buying opportunities
    - Seasonal shopping recommendations
    - Overall spending optimization tips
    
    Args:
        comparisons (list): List of price comparison dictionaries
        
    Returns:
        list: List of shopping suggestion dictionaries with advice and impact
    """
    suggestions = []
    
    if not comparisons:
        return [{
            'type': 'success',
            'title': '✅ Excellent Shopping!',
            'description': "You're getting competitive prices on your groceries. Keep up the great work!",
            'action': 'continue'
        }]
    
    # Group by suggestion type and merchant
    excellent_savings = [c for c in comparisons if c['suggestion_type'] == 'excellent']
    significant_savings = [c for c in comparisons if c['suggestion_type'] == 'significant']
    moderate_savings = [c for c in comparisons if c['suggestion_type'] == 'moderate']
    
    # Calculate total potential savings
    total_potential_savings = sum(c['savings_amount'] for c in comparisons)
    
    # Group savings by merchant for smarter recommendations
    merchant_savings = {}
    for c in comparisons:
        merchant = c['best_merchant']
        if merchant not in merchant_savings:
            merchant_savings[merchant] = {'items': [], 'total_savings': 0, 'avg_percentage': 0}
        merchant_savings[merchant]['items'].append(c)
        merchant_savings[merchant]['total_savings'] += c['savings_amount']
    
    # Calculate average savings percentage per merchant
    for merchant in merchant_savings:
        items = merchant_savings[merchant]['items']
        merchant_savings[merchant]['avg_percentage'] = sum(item['savings_percentage'] for item in items) / len(items)
    
    # Sort merchants by total potential savings
    top_merchants = sorted(merchant_savings.items(), 
                          key=lambda x: x[1]['total_savings'], 
                          reverse=True)[:3]
    
    if excellent_savings or significant_savings:
        best_merchant = top_merchants[0][0] if top_merchants else None
        best_merchant_data = top_merchants[0][1] if top_merchants else None
        
        if best_merchant_data:
            suggestions.append({
                'type': 'high_impact',
                'title': f'🚨 Major Savings Opportunity: ${best_merchant_data["total_savings"]:.2f}',
                'description': f"Shopping at {best_merchant} could save you ${best_merchant_data['total_savings']:.2f} on {len(best_merchant_data['items'])} items (avg {best_merchant_data['avg_percentage']:.1f}% off)",
                'items': [f"{item['item_name']}: Save ${item['savings_amount']:.2f}" for item in best_merchant_data['items']],
                'action': 'switch_store',
                'merchant': best_merchant
            })
    
    # Multi-store optimization suggestion
    if len(top_merchants) > 1 and total_potential_savings > 10:
        suggestions.append({
            'type': 'optimization',
            'title': f'🎯 Multi-Store Strategy: Save ${total_potential_savings:.2f}',
            'description': f"Optimize across {len(top_merchants)} stores for maximum savings",
            'items': [f"{merchant}: ${data['total_savings']:.2f} on {len(data['items'])} items" for merchant, data in top_merchants],
            'action': 'multi_store',
            'total_savings': total_potential_savings
        })
    
    if moderate_savings and not excellent_savings and not significant_savings:
        suggestions.append({
            'type': 'moderate_impact',
            'title': f'💰 Small Savings Add Up: ${total_potential_savings:.2f}',
            'description': f"While individual savings are small, you could save ${total_potential_savings:.2f} total",
            'items': [f"{c['item_name']}: ${c['savings_amount']:.2f} at {c['best_merchant']}" for c in moderate_savings[:3]],
            'action': 'consider'
        })
    
    # Add market intelligence
    high_reliability_deals = [c for c in comparisons if c.get('reliability_score', 0) > 80]
    if high_reliability_deals:
        suggestions.append({
            'type': 'intelligence',
            'title': '📊 Market Intelligence',
            'description': f"Based on reliable price data, these {len(high_reliability_deals)} deals are consistently good",
            'items': [f"{c['item_name']} at {c['best_merchant']} (reliable)" for c in high_reliability_deals[:3]],
            'action': 'trust'
        })
    if len(comparisons) > 0:
        avg_savings_percentage = sum(c['savings_percentage'] for c in comparisons) / len(comparisons)
        if avg_savings_percentage > 15:
            suggestions.append({
                'type': 'store_recommendation',
                'title': '🏪 Store Recommendation',
                'description': f"You could save an average of {avg_savings_percentage:.1f}% by diversifying where you shop for groceries",
                'items': []
            })
    
    return suggestions

# Error handlers to ensure API endpoints return JSON
@app.errorhandler(404)
def not_found_error(error):
    """
    Handle 404 Not Found errors.
    
    Returns a JSON error response for API endpoints and
    redirects to home page for web requests.
    """
    # Return JSON for API calls, HTML for page requests
    if request.path.startswith('/') and request.path != '/' and (request.is_json or 'application/json' in request.headers.get('Accept', '')):
        return jsonify({'error': 'Endpoint not found', 'path': request.path}), 404
    return render_template('index.html'), 404  # Redirect to main page instead of 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 Internal Server Error.
    
    Returns a JSON error response for server-side errors.
    """
    print(f"Internal Server Error: {error}")
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

@app.errorhandler(400)
def bad_request_error(error):
    """
    Handle 400 Bad Request errors.
    
    Returns a JSON error response for malformed requests.
    """
    print(f"Bad Request Error: {error}")
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.route('/')
def index():
    """
    Serve the main web interface.
    
    Returns the main HTML page with the expense scanner interface.
    """
    return render_template('index.html')

@app.route('/csrf-token')
def csrf_token():
    """
    Generate and return a CSRF token for form protection.
    
    Returns:
        JSON response containing a fresh CSRF token
    """
    from flask_wtf.csrf import generate_csrf
    return jsonify({'csrf_token': generate_csrf()})

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Main file upload endpoint for processing receipt images.
    
    Processes uploaded receipt images through OCR, expense categorization,
    grocery item extraction, price comparisons, and location-based features.
    
    Workflow:
    1. Validate and save uploaded image file
    2. Extract text using OCR (multiple methods with fallbacks)
    3. Categorize expense and extract merchant information
    4. For grocery receipts: extract items and compare prices
    5. Find nearby stores based on receipt address
    6. Save all data to database
    7. Return comprehensive expense analysis
    
    Returns:
        JSON response with expense data, price comparisons, and nearby stores
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save file
        filename = f"receipt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from image
            raw_text = extract_text_from_image(filepath)
            
            # Categorize expense
            expense_data = categorize_expense_free(raw_text)
            print(f"📄 Receipt processed - Merchant: '{expense_data['merchant']}', Category: '{expense_data['category']}', Amount: ${expense_data['amount']}")
            
            # If it's a grocery expense, extract items and get price comparisons
            grocery_items = []
            price_comparisons = []
            shopping_suggestions = []
            nearby_stores = []
            
            # Find nearby stores based on receipt address
            if expense_data.get('address_info'):
                nearby_stores = find_stores_by_address(expense_data['address_info'])
                print(f"🏬 Found {len(nearby_stores)} nearby stores based on receipt address")
            
            if expense_data['category'] == 'Groceries':
                grocery_items = extract_grocery_items(raw_text)
                
                # Get price comparisons with other merchants
                current_merchant = expense_data['merchant']
                price_comparisons = get_price_comparisons(grocery_items, current_merchant)
                
                # Generate shopping suggestions
                shopping_suggestions = generate_shopping_suggestions(price_comparisons)
            
            # Save to database
            conn = sqlite3.connect('expenses.db')
            c = conn.cursor()
            c.execute("""INSERT INTO expenses (amount, category, merchant, date, raw_text)
                        VALUES (?, ?, ?, ?, ?)""",
                     (expense_data['amount'], expense_data['category'], 
                      expense_data['merchant'], datetime.now().strftime('%Y-%m-%d'),
                      raw_text))
            expense_id = c.lastrowid
            conn.commit()
            conn.close()
            
            # Save grocery items and price comparisons if it's a grocery expense
            if grocery_items:
                save_grocery_items(expense_id, grocery_items)
                # Save price history for future comparisons
                save_price_history(grocery_items, expense_data['merchant'])
            if price_comparisons:
                save_price_comparisons(expense_id, price_comparisons)
            
            response_data = {
                'success': True,
                'id': expense_id,
                'amount': expense_data['amount'],
                'category': expense_data['category'],
                'merchant': expense_data['merchant'],
                'raw_text': raw_text,
                'confidence': expense_data.get('confidence', 0.8)
            }
            
            # Add grocery items and price comparisons if it's a grocery expense
            if expense_data['category'] == 'Groceries':
                response_data['grocery_items'] = grocery_items
                response_data['price_comparisons'] = price_comparisons
                response_data['shopping_suggestions'] = shopping_suggestions
            
            # Add nearby stores and address information
            if nearby_stores:
                response_data['nearby_stores'] = nearby_stores
            if expense_data.get('address_info'):
                response_data['store_address'] = expense_data['address_info']
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

@app.route('/expenses')
def get_expenses():
    """
    API endpoint to retrieve user's expense history.
    
    Returns a chronological list of all recorded expenses with
    basic information for display in the user interface.
    
    Returns:
        JSON array of expense records
    """
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("""SELECT id, amount, category, merchant, date, created_at 
                FROM expenses ORDER BY created_at DESC LIMIT 50""")
    expenses = []
    for row in c.fetchall():
        expenses.append({
            'id': row[0],
            'amount': row[1],
            'category': row[2],
            'merchant': row[3],
            'date': row[4],
            'created_at': row[5]
        })
    conn.close()
    return jsonify(expenses)

@app.route('/stats')
def get_stats():
    """
    API endpoint to get expense statistics and spending analysis.
    
    Calculates and returns:
    - Total spending amount
    - Top spending categories
    - Category breakdown with amounts
    
    Returns:
        JSON object with total spending and category statistics
    """
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    
    # Total expenses
    c.execute("SELECT SUM(amount) FROM expenses")
    total = c.fetchone()[0] or 0
    
    # Category breakdown
    c.execute("""SELECT category, SUM(amount), COUNT(*) 
                FROM expenses GROUP BY category ORDER BY SUM(amount) DESC""")
    categories = []
    for row in c.fetchall():
        categories.append({
            'category': row[0],
            'amount': row[1],
            'count': row[2]
        })
    
    conn.close()
    return jsonify({
        'total': total,
        'categories': categories
    })

@app.route('/profile', methods=['GET', 'POST'])
def user_profile():
    """
    Handle user profile management (GET and POST).
    
    GET: Retrieve current user profile data
    POST: Save updated user profile information including dietary
          preferences, allergies, and cooking skills
    
    Returns:
        GET: JSON object with current profile data
        POST: JSON success/error response
    """
    if request.method == 'GET':
        profile = get_user_profile()
        return jsonify(profile or {})
    
    elif request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        try:
            conn = sqlite3.connect('expenses.db')
            c = conn.cursor()
            
            # Update or insert user profile
            c.execute("""INSERT OR REPLACE INTO user_profile 
                        (id, diet_type, allergies, cooking_skill, preferred_cuisines, updated_at)
                        VALUES ((SELECT id FROM user_profile LIMIT 1), ?, ?, ?, ?, ?)""",
                     (data.get('diet_type', ''),
                      data.get('allergies', ''),
                      data.get('cooking_skill', 'beginner'),
                      data.get('preferred_cuisines', ''),
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': 'Profile updated successfully'})
            
        except Exception as e:
            return jsonify({'error': f'Failed to update profile: {str(e)}'}), 500

@app.route('/update-location', methods=['POST'])
def update_location():
    """
    Update user's location for location-aware features.
    
    Accepts latitude/longitude coordinates and updates the user profile
    to enable location-based price comparisons and nearby store features.
    
    Returns:
        JSON success/error response with location update status
    """
    """Update user location for location-aware features"""
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        address = data.get('address', '')
        
        if not latitude or not longitude:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        # Check if profile exists
        c.execute('SELECT id FROM user_profile ORDER BY id DESC LIMIT 1')
        existing = c.fetchone()
        
        if existing:
            # Update existing profile
            c.execute('''UPDATE user_profile SET 
                        latitude = ?, longitude = ?, address = ?, 
                        location_updated_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?''',
                     (latitude, longitude, address, existing[0]))
        else:
            # Create new profile with location
            c.execute('''INSERT INTO user_profile 
                        (latitude, longitude, address, location_updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)''',
                     (latitude, longitude, address))
        
        conn.commit()
        conn.close()
        
        # Find nearby stores
        nearby_stores = find_nearby_stores(latitude, longitude)
        
        return jsonify({
            'success': True, 
            'message': 'Location updated successfully',
            'nearby_stores': nearby_stores[:5],  # Top 5 nearest stores
            'total_nearby_stores': len(nearby_stores)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to update location: {str(e)}'}), 500

@app.route('/price-comparisons/<int:expense_id>')
def get_price_comparisons_for_expense(expense_id):
    """
    Get price comparisons for a specific expense record.
    
    Retrieves saved price comparison data for a particular expense,
    showing potential savings and alternative store recommendations.
    
    Args:
        expense_id (int): Database ID of the expense record
        
    Returns:
        JSON array of price comparison data
    """
    """Get price comparisons for a specific grocery expense"""
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        c.execute("""SELECT item_name, current_price, current_merchant, best_price, 
                           best_merchant, savings_amount, savings_percentage, suggestion
                    FROM price_comparisons 
                    WHERE expense_id = ? 
                    ORDER BY savings_amount DESC""",
                 (expense_id,))
        
        comparisons = []
        for row in c.fetchall():
            comparisons.append({
                'item_name': row[0],
                'current_price': row[1],
                'current_merchant': row[2],
                'best_price': row[3],
                'best_merchant': row[4],
                'savings_amount': row[5],
                'savings_percentage': row[6],
                'suggestion': row[7]
            })
        
        conn.close()
        return jsonify(comparisons)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get price comparisons: {str(e)}'}), 500

@app.route('/merchant-trends/<merchant_name>')
def get_merchant_trends(merchant_name):
    """
    Get price trend data for a specific merchant.
    
    Analyzes historical pricing patterns for a merchant over
    a specified time period (default 30 days).
    
    Args:
        merchant_name (str): Name of the merchant to analyze
        
    Returns:
        JSON object with price trend analysis
    """
    """Get price trends for a specific merchant"""
    try:
        days = request.args.get('days', 30, type=int)
        trends = get_merchant_price_trends(merchant_name, days)
        return jsonify(trends)
    except Exception as e:
        return jsonify({'error': f'Failed to get merchant trends: {str(e)}'}), 500

@app.route('/price-insights')
def get_price_insights():
    """
    Get overall price insights and spending recommendations.
    
    Analyzes user's spending patterns to provide:
    - Top savings opportunities by item
    - Best merchants for various categories
    - Monthly potential savings estimates
    - Personalized shopping recommendations
    
    Returns:
        JSON object with comprehensive price insights and recommendations
    """
    """Get overall price insights and recommendations"""
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        # Get top savings opportunities
        c.execute("""SELECT item_name, AVG(savings_amount) as avg_savings,
                           COUNT(*) as frequency,
                           best_merchant as recommended_merchant
                    FROM price_comparisons
                    WHERE savings_amount > 1.0
                    GROUP BY item_name, best_merchant
                    HAVING frequency >= 2
                    ORDER BY avg_savings DESC
                    LIMIT 10""")
        
        top_savings = []
        for row in c.fetchall():
            top_savings.append({
                'item_name': row[0],
                'avg_savings': row[1],
                'frequency': row[2],
                'recommended_merchant': row[3]
            })
        
        # Get best merchants overall
        c.execute("""SELECT best_merchant, COUNT(*) as deal_count,
                           AVG(savings_percentage) as avg_savings_pct
                    FROM price_comparisons
                    GROUP BY best_merchant
                    HAVING deal_count >= 3
                    ORDER BY avg_savings_pct DESC
                    LIMIT 5""")
        
        best_merchants = []
        for row in c.fetchall():
            best_merchants.append({
                'merchant': row[0],
                'deal_count': row[1],
                'avg_savings_percentage': row[2]
            })
        
        # Calculate total potential savings
        c.execute("SELECT SUM(savings_amount) FROM price_comparisons WHERE created_at >= date('now', '-30 days')")
        monthly_potential_savings = c.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'top_savings_opportunities': top_savings,
            'best_merchants_for_deals': best_merchants,
            'monthly_potential_savings': monthly_potential_savings,
            'insights_period': '30 days'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get price insights: {str(e)}'}), 500

@app.route('/add-sample-price-data', methods=['POST'])
def add_sample_price_data():
    """
    Development endpoint to add sample price data for testing.
    
    Populates the database with sample price history data to
    demonstrate price comparison functionality during development.
    
    Returns:
        JSON success/error response
    """
    """Add sample price data for demonstration (development only)"""
    try:
        sample_data = [
            # Walmart prices
            {'merchant': 'WALMART', 'item': 'Banana', 'price': 1.28, 'quantity': 2.5, 'unit': 'lb'},
            {'merchant': 'WALMART', 'item': 'Chicken', 'price': 8.99, 'quantity': 2.0, 'unit': 'lb'},
            {'merchant': 'WALMART', 'item': 'Milk', 'price': 3.78, 'quantity': 1, 'unit': 'each'},
            {'merchant': 'WALMART', 'item': 'Bread', 'price': 2.48, 'quantity': 1, 'unit': 'each'},
            
            # Target prices
            {'merchant': 'TARGET', 'item': 'Banana', 'price': 1.49, 'quantity': 2.5, 'unit': 'lb'},
            {'merchant': 'TARGET', 'item': 'Chicken', 'price': 9.99, 'quantity': 2.0, 'unit': 'lb'},
            {'merchant': 'TARGET', 'item': 'Milk', 'price': 3.99, 'quantity': 1, 'unit': 'each'},
            {'merchant': 'TARGET', 'item': 'Bread', 'price': 2.79, 'quantity': 1, 'unit': 'each'},
            
            # Whole Foods prices
            {'merchant': 'WHOLE FOODS', 'item': 'Banana', 'price': 1.79, 'quantity': 2.5, 'unit': 'lb'},
            {'merchant': 'WHOLE FOODS', 'item': 'Chicken', 'price': 12.99, 'quantity': 2.0, 'unit': 'lb'},
            {'merchant': 'WHOLE FOODS', 'item': 'Milk', 'price': 4.99, 'quantity': 1, 'unit': 'each'},
            {'merchant': 'WHOLE FOODS', 'item': 'Bread', 'price': 3.99, 'quantity': 1, 'unit': 'each'},
        ]
        
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        for data in sample_data:
            price_per_unit = data['price'] / data['quantity'] if data['quantity'] > 0 else data['price']
            
            c.execute("""INSERT INTO price_history 
                        (merchant_name, item_name, price, quantity, unit_type, 
                         price_per_unit, location, date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                     (data['merchant'], data['item'], data['price'],
                      str(data['quantity']), data['unit'], price_per_unit,
                      'Sample Location', datetime.now().strftime('%Y-%m-%d')))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'Added {len(sample_data)} sample price records'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to add sample data: {str(e)}'}), 500

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two coordinates using Haversine formula.
    
    Computes the great-circle distance between two points on Earth
    given their latitude and longitude coordinates.
    
    Args:
        lat1, lon1 (float): Latitude and longitude of first point
        lat2, lon2 (float): Latitude and longitude of second point
        
    Returns:
        float: Distance in miles
    """
    import math
    
    R = 3959  # Earth's radius in miles
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return round(distance, 2)

def find_stores_by_address(address_info, radius_miles=15):
    """
    Find stores near a specific address using geocoding or city matching.
    
    Uses the address information extracted from a receipt to locate
    nearby stores for price comparison and shopping convenience.
    
    Args:
        address_info (dict): Address information with city/state data
        radius_miles (int): Search radius in miles (default: 15)
        
    Returns:
        list: List of nearby store dictionaries
    """
    try:
        # If we have specific location coordinates, use them
        if address_info.get('city') and address_info.get('state'):
            return find_stores_by_city(address_info['city'], address_info['state'], radius_miles)
        
        return []
        
    except Exception as e:
        print(f"Error finding stores by address: {e}")
        return []

def find_stores_by_city(city, state, radius_miles=15):
    """
    Find stores in the same city and state.
    
    Searches the local stores database for all stores located
    in the specified city and state.
    
    Args:
        city (str): City name
        state (str): State or province abbreviation
        radius_miles (int): Not used in this function, kept for compatibility
        
    Returns:
        list: List of store dictionaries in the specified city/state
    """
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        # Look for stores in the same city and state
        c.execute('''SELECT id, store_name, chain_name, address, city, state, 
                           latitude, longitude, phone, store_hours, store_type
                    FROM local_stores 
                    WHERE UPPER(city) = UPPER(?) AND UPPER(state) = UPPER(?)''',
                 (city, state))
        
        stores = c.fetchall()
        nearby_stores = []
        
        for store in stores:
            store_id, name, chain, address, store_city, store_state, lat, lon, phone, hours, store_type = store
            nearby_stores.append({
                'id': store_id,
                'name': name,
                'chain': chain,
                'address': address,
                'city': store_city,
                'state': store_state,
                'latitude': lat,
                'longitude': lon,
                'phone': phone,
                'hours': hours,
                'type': store_type,
                'distance': 0  # Same city, so close distance
            })
        
        print(f"🏬 Found {len(nearby_stores)} stores in {city}, {state}")
        conn.close()
        return nearby_stores
        
    except Exception as e:
        print(f"Error finding stores by city: {e}")
        return []

def find_nearby_stores(user_lat, user_lon, radius_miles=10):
    """Find stores within radius of user location"""
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        c.execute('''SELECT id, store_name, chain_name, address, city, state, 
                           latitude, longitude, phone, store_hours, store_type
                    FROM local_stores 
                    WHERE latitude IS NOT NULL AND longitude IS NOT NULL''')
        
        all_stores = c.fetchall()
        nearby_stores = []
        
        for store in all_stores:
            store_id, name, chain, address, city, state, lat, lon, phone, hours, store_type = store
            distance = calculate_distance(user_lat, user_lon, lat, lon)
            
            if distance <= radius_miles:
                nearby_stores.append({
                    'id': store_id,
                    'name': name,
                    'chain': chain,
                    'address': address,
                    'city': city,
                    'state': state,
                    'latitude': lat,
                    'longitude': lon,
                    'phone': phone,
                    'hours': hours,
                    'type': store_type,
                    'distance': distance
                })
        
        # Sort by distance
        nearby_stores.sort(key=lambda x: x['distance'])
        conn.close()
        return nearby_stores
        
    except Exception as e:
        print(f"Error finding nearby stores: {e}")
        return []

def populate_sample_stores():
    """
    Add sample stores for testing location-aware features.
    
    Populates database with sample store locations across different cities
    to enable testing of location-based features and price comparisons.
    
    Creates stores in:
    - Toronto, ON (Canadian stores)
    - San Jose, CA and Los Angeles, CA (US stores)
    
    Returns:
        None
    """
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        sample_stores = [
            # Toronto/GTA stores (for Loblaws receipt testing)
            ('Loblaws Toronto Lakeshore', 'Loblaws', '500 Lakeshore Blvd W, Toronto, ON M5V 1A1', 'Toronto', 'ON', 'M5V 1A1', 43.6331, -79.3961, '(416) 366-5036', '7:00 AM - 11:00 PM', 'grocery'),
            ('Metro', 'Metro', '33 King St W, Toronto, ON M5H 1B4', 'Toronto', 'ON', 'M5H 1B4', 43.6477, -79.3798, '(416) 593-8888', '7:00 AM - 10:00 PM', 'grocery'),
            ('No Frills', 'Loblaws', '25 The Esplanade, Toronto, ON M5E 1S6', 'Toronto', 'ON', 'M5E 1S6', 43.6463, -79.3712, '(416) 368-6603', '7:00 AM - 11:00 PM', 'grocery'),
            ('Sobeys', 'Sobeys', '87 Front St E, Toronto, ON M5E 1C3', 'Toronto', 'ON', 'M5E 1C3', 43.6475, -79.3711, '(416) 368-2627', '7:00 AM - 11:00 PM', 'grocery'),
            ('FreshCo', 'Sobeys', '629 Lake Shore Blvd W, Toronto, ON M5V 1A9', 'Toronto', 'ON', 'M5V 1A9', 43.6314, -79.4014, '(416) 260-5400', '7:00 AM - 11:00 PM', 'grocery'),
            
            # Bay Area stores  
            ('Walmart Supercenter', 'Walmart', '3980 Automation Way, San Jose, CA 95131', 'San Jose', 'CA', '95131', 37.4135, -121.9530, '(408) 324-0200', '6:00 AM - 11:00 PM', 'grocery'),
            ('Target', 'Target', '777 Story Rd, San Jose, CA 95122', 'San Jose', 'CA', '95122', 37.3230, -121.8525, '(408) 281-4800', '8:00 AM - 10:00 PM', 'grocery'),
            ('Safeway', 'Safeway', '1071 E Capitol Expy, San Jose, CA 95121', 'San Jose', 'CA', '95121', 37.3074, -121.8370, '(408) 274-9120', '5:00 AM - 12:00 AM', 'grocery'),
            ('Whole Foods Market', 'Whole Foods', '3245 Stevens Creek Blvd, San Jose, CA 95117', 'San Jose', 'CA', '95117', 37.3230, -121.9530, '(408) 296-9401', '7:00 AM - 10:00 PM', 'grocery'),
            ('Costco Wholesale', 'Costco', '1709 Automation Pkwy, San Jose, CA 95131', 'San Jose', 'CA', '95131', 37.4055, -121.9501, '(408) 324-0347', '10:00 AM - 8:30 PM', 'wholesale'),
            
            # Los Angeles stores
            ('Walmart Neighborhood Market', 'Walmart', '2930 Los Feliz Blvd, Los Angeles, CA 90039', 'Los Angeles', 'CA', '90039', 34.1184, -118.2815, '(323) 660-3171', '6:00 AM - 11:00 PM', 'grocery'),
            ('Target', 'Target', '7100 Santa Monica Blvd, West Hollywood, CA 90046', 'West Hollywood', 'CA', '90046', 34.0983, -118.3432, '(323) 785-7400', '7:00 AM - 11:00 PM', 'grocery'),
            ('Ralphs', 'Kroger', '645 W 9th St, Los Angeles, CA 90015', 'Los Angeles', 'CA', '90015', 34.0417, -118.2608, '(213) 486-7315', '6:00 AM - 12:00 AM', 'grocery'),
            ('Whole Foods Market', 'Whole Foods', '239 S La Brea Ave, Los Angeles, CA 90036', 'Los Angeles', 'CA', '90036', 34.0734, -118.3440, '(323) 936-7800', '7:00 AM - 10:00 PM', 'grocery'),
        ]
        
        for store_data in sample_stores:
            c.execute('''INSERT OR IGNORE INTO local_stores 
                        (store_name, chain_name, address, city, state, zip_code, 
                         latitude, longitude, phone, store_hours, store_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', store_data)
        
        conn.commit()
        conn.close()
        print("✅ Sample stores populated successfully!")
        
    except Exception as e:
        print(f"Error populating sample stores: {e}")

def populate_sample_inventory():
    """
    Add sample inventory data for local stores.
    
    Populates database with sample grocery item inventory and varying
    prices across different stores to enable price comparison functionality.
    
    Creates inventory for common items like:
    - Rice, Bread, Milk, Eggs
    - Chicken Breast, Ground Beef
    - Apples, Banana
    
    Returns:
        None
    """
    try:
        conn = sqlite3.connect('expenses.db')
        c = conn.cursor()
        
        # Get store IDs
        c.execute('SELECT id, chain_name FROM local_stores')
        stores = c.fetchall()
        
        # Sample inventory with chain-specific pricing
        inventory_data = {
            'Walmart': [
                ('Banana', 'Produce', 0.58, 'lb', 50),
                ('Chicken Breast', 'Meat', 4.27, 'lb', 20),
                ('Milk', 'Dairy', 3.64, 'gallon', 30),
                ('Bread', 'Bakery', 1.00, 'loaf', 25),
                ('Eggs', 'Dairy', 2.57, 'dozen', 40),
                ('Rice', 'Pantry', 1.98, 'bag', 15),
                ('Apples', 'Produce', 1.47, 'lb', 35),
                ('Ground Beef', 'Meat', 4.97, 'lb', 18),
            ],
            'Target': [
                ('Banana', 'Produce', 0.64, 'lb', 45),
                ('Chicken Breast', 'Meat', 4.99, 'lb', 15),
                ('Milk', 'Dairy', 3.79, 'gallon', 25),
                ('Bread', 'Bakery', 2.49, 'loaf', 20),
                ('Eggs', 'Dairy', 2.89, 'dozen', 35),
                ('Rice', 'Pantry', 2.29, 'bag', 12),
                ('Apples', 'Produce', 1.69, 'lb', 30),
                ('Ground Beef', 'Meat', 5.49, 'lb', 14),
            ],
            'Whole Foods': [
                ('Banana', 'Produce', 0.79, 'lb', 40),
                ('Chicken Breast', 'Meat', 8.99, 'lb', 10),
                ('Milk', 'Dairy', 4.99, 'gallon', 20),
                ('Bread', 'Bakery', 4.99, 'loaf', 15),
                ('Eggs', 'Dairy', 4.49, 'dozen', 25),
                ('Rice', 'Pantry', 3.99, 'bag', 8),
                ('Apples', 'Produce', 2.49, 'lb', 25),
                ('Ground Beef', 'Meat', 7.99, 'lb', 8),
            ],
            'Safeway': [
                ('Banana', 'Produce', 0.69, 'lb', 40),
                ('Chicken Breast', 'Meat', 5.49, 'lb', 18),
                ('Milk', 'Dairy', 3.99, 'gallon', 28),
                ('Bread', 'Bakery', 2.99, 'loaf', 22),
                ('Eggs', 'Dairy', 3.29, 'dozen', 32),
                ('Rice', 'Pantry', 2.49, 'bag', 14),
                ('Apples', 'Produce', 1.79, 'lb', 28),
                ('Ground Beef', 'Meat', 5.99, 'lb', 16),
            ],
            'Kroger': [
                ('Banana', 'Produce', 0.62, 'lb', 42),
                ('Chicken Breast', 'Meat', 4.79, 'lb', 16),
                ('Milk', 'Dairy', 3.49, 'gallon', 26),
                ('Bread', 'Bakery', 2.79, 'loaf', 18),
                ('Eggs', 'Dairy', 2.99, 'dozen', 30),
                ('Rice', 'Pantry', 2.19, 'bag', 11),
                ('Apples', 'Produce', 1.59, 'lb', 32),
                ('Ground Beef', 'Meat', 5.29, 'lb', 15),
            ]
        }
        
        for store_id, chain_name in stores:
            if chain_name in inventory_data:
                for item_name, category, price, unit, stock in inventory_data[chain_name]:
                    c.execute('''INSERT OR IGNORE INTO store_inventory 
                                (store_id, item_name, category, price_per_unit, unit_type, 
                                 stock_quantity, last_updated)
                                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                             (store_id, item_name, category, price, unit, stock))
        
        conn.commit()
        conn.close()
        print("✅ Sample inventory populated successfully!")
        
    except Exception as e:
        print(f"Error populating sample inventory: {e}")

def validate_environment():
    """
    Validate required environment variables are set.
    
    Checks that all required environment variables are properly configured
    for the expense scanner application to function correctly.
    
    Validates:
    - Hugging Face API token
    - OCR Space API key
    - Flask secret key
    
    Returns:
        bool: True if all validations pass, False otherwise
    """
    required_vars = {
        'HUGGINGFACE_API_TOKEN': 'Get your free token at: https://huggingface.co/settings/tokens',
        'FLASK_SECRET_KEY': 'Generate with: python -c "import secrets; print(secrets.token_hex(32))"'
    }
    
    missing_vars = []
    for var, help_text in required_vars.items():
        value = os.getenv(var)
        if not value or value == f'{var.lower()}_here' or 'your-' in value or 'change' in value:
            missing_vars.append(f"  ❌ {var}: {help_text}")
    
    if missing_vars:
        print("\n" + "="*60)
        print("🔑 CONFIGURATION ERROR: Missing required environment variables!")
        print("="*60)
        print("\n".join(missing_vars))
        print(f"\n📋 Setup instructions:")
        print(f"  1. Copy .env.example to .env:  cp .env.example .env")
        print(f"  2. Edit .env file with your actual API keys")
        print(f"  3. See README.md for detailed setup instructions")
        print("="*60)
        return False
    
    print("✅ Environment variables validated successfully!")
    return True

if __name__ == '__main__':
    # Validate environment before starting
    if not validate_environment():
        print("❌ Please fix your .env configuration before running the app.")
        exit(1)
    
    init_db()
    populate_sample_stores()
    populate_sample_inventory()
    print("🚀 AI Expense Scanner starting...")
    print("📊 Visit http://localhost:5000 to start scanning receipts!")
    
    # Check if we're in production
    flask_env = os.getenv('FLASK_ENV', 'development')
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)