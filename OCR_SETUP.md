# OCR Setup Instructions

## The Problem
Currently, the receipt scanner is using fake demo data instead of actually reading receipts because the OCR API key is invalid.

## Get a Free OCR.space API Key

1. **Visit**: https://ocr.space/ocrapi
2. **Sign up** for a free account
3. **Get your API key** (free tier: 25,000 requests/month)
4. **Update your .env file**:
   ```
   OCR_SPACE_API_KEY=your_actual_api_key_here
   ```

## Alternative: Use Local OCR (No API needed)

Install Tesseract OCR locally:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

Then install Python wrapper:
```bash
pip install pytesseract
```

## Why This Matters

- **Currently**: Using hardcoded "WALMART SUPERCENTER" text
- **With real OCR**: Will read actual store names from your receipts
- **Result**: Accurate merchant detection from real receipt images

The system now trusts OCR to read shop names directly from receipts:
- OCR reads: "WAL*MART SUPER CENTER #1234" → Merchant: "WAL*MART SUPER CENTER #1234"
- OCR reads: "Target Store Number 5678" → Merchant: "TARGET STORE NUMBER 5678"  
- OCR reads: "SAFEWAY INC." → Merchant: "SAFEWAY INC."

The merchant name is preserved exactly as OCR reads it from your receipt!