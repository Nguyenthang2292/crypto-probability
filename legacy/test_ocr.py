
import sys
import os

# Add the project directory to sys.path
project_dir = r"d:\NGUYEN QUANG THANG\Probability projects\crypto-probability-"
sys.path.append(project_dir)

from portfolio_ui import extract_text_from_image, parse_portfolio_from_text
import cv2
import numpy as np

def test_ocr():
    image_path = r"C:/Users/Admin/.gemini/antigravity/brain/4ca7b993-203a-4bf1-aff5-5de6943c1a86/uploaded_image_1763626794635.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return

    print(f"Processing image: {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image")
        return

    # Extract text
    print("Extracting text...")
    text = extract_text_from_image(img)
    print("-" * 40)
    print("RAW OCR TEXT:")
    print(text)
    print("-" * 40)

    # Parse
    print("Parsing portfolio...")
    positions = parse_portfolio_from_text(text)
    
    print(f"Found {len(positions)} positions:")
    for pos in positions:
        print(pos)

if __name__ == "__main__":
    test_ocr()
