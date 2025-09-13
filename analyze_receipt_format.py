#!/usr/bin/env python3

import sys
import re
sys.path.append('/Users/selamberhane/Projects/ai-expense-scanner')

from expense_scanner_python import extract_text_from_image

def analyze_receipt_format():
    """Analyze the actual receipt format to understand price patterns"""
    
    image_path = '/Users/selamberhane/Downloads/IMG20250826171827.jpg'
    raw_text = extract_text_from_image(image_path)
    
    print("🧪 Analyzing receipt format for price patterns...\n")
    
    lines = raw_text.split('\n')
    
    print("📄 Full receipt text with line numbers:")
    print("=" * 80)
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: '{line}'")
    print("=" * 80)
    
    print("\n💰 Looking for price patterns...")
    
    # Look for various price patterns
    price_patterns = [
        r'\$(\d+\.\d{2})',           # $1.29
        r'(\d+\.\d{2})\s*$',        # 1.29 at end of line
        r'(\d+\.\d{2})\s*T',        # 1.29 T (taxable)
        r'(\d+\.\d{2})\s*N',        # 1.29 N (non-taxable)
        r'(\d+\.\d{2})\s*[A-Z]',    # 1.29 followed by letter
        r'(\d+\.\d{2})\s+\w+',      # 1.29 followed by word
    ]
    
    for pattern_name, pattern in [
        ('Dollar sign pattern', r'\$(\d+\.\d{2})'),
        ('End of line pattern', r'(\d+\.\d{2})\s*$'),
        ('With T suffix', r'(\d+\.\d{2})\s*T'),
        ('With N suffix', r'(\d+\.\d{2})\s*N'),
        ('With letter suffix', r'(\d+\.\d{2})\s*[A-Z]'),
        ('With word after', r'(\d+\.\d{2})\s+\w+'),
    ]:
        print(f"\n{pattern_name}: {pattern}")
        matches = []
        for i, line in enumerate(lines, 1):
            found = re.findall(pattern, line)
            if found:
                matches.extend([(i, line, price) for price in found])
        
        if matches:
            print(f"  Found {len(matches)} matches:")
            for line_num, line_text, price in matches[:10]:  # Show first 10
                print(f"    Line {line_num}: '{line_text.strip()}' → ${price}")
        else:
            print("  No matches found")
    
    print("\n🛒 Looking for grocery item lines with prices...")
    grocery_keywords = ['tuna', 'rice', 'tomato', 'pepper', 'lemon', 'avocado', 'beans', 'milk', 'bread', 'cheese']
    
    for keyword in grocery_keywords:
        for i, line in enumerate(lines, 1):
            if keyword.lower() in line.lower():
                print(f"  Line {i}: '{line}' (contains '{keyword}')")
                # Look for prices in this line
                prices = re.findall(r'(\d+\.\d{2})', line)
                if prices:
                    print(f"    → Potential prices: {prices}")
                else:
                    print(f"    → No prices found")

if __name__ == '__main__':
    analyze_receipt_format()