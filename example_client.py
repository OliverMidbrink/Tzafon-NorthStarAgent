#!/usr/bin/env python3
"""
Example client for UI-TARS HTTP API
Shows how to send images and get analysis results.
"""

import os
import requests
import sys
from pathlib import Path

# Default server URL
SERVER_URL = "http://localhost:8000"

# Load API key from environment or file
def load_api_key():
    api_key = os.getenv("GPU_API_KEY")
    if api_key:
        return api_key
    
    try:
        with open(".api_key.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "ui-tars-example-key"

API_KEY = load_api_key()

def test_health():
    """Test if server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        print(f"Health check: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Server not accessible: {e}")
        return False

def analyze_image(image_path: str, query: str, max_tokens: int = 500):
    """Analyze an image with a custom query"""
    
    if not Path(image_path).exists():
        print(f"Error: Image {image_path} not found")
        return None
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    with open(image_path, 'rb') as f:
        files = {"file": f}
        data = {
            "query": query,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{SERVER_URL}/analyze",
                headers=headers,
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Analysis result: {result['result']}")
                return result
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

def describe_image(image_path: str, max_tokens: int = 500):
    """Get a detailed description of an image"""
    
    if not Path(image_path).exists():
        print(f"Error: Image {image_path} not found")
        return None
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    with open(image_path, 'rb') as f:
        files = {"file": f}
        data = {"max_tokens": max_tokens}
        
        try:
            response = requests.post(
                f"{SERVER_URL}/describe",
                headers=headers,
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Description: {result['result']}")
                return result
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

def click_analysis(image_path: str, x: int, y: int, max_tokens: int = 500):
    """Analyze what happens when clicking coordinates"""
    
    if not Path(image_path).exists():
        print(f"Error: Image {image_path} not found")
        return None
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    with open(image_path, 'rb') as f:
        files = {"file": f}
        data = {
            "x": x,
            "y": y,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{SERVER_URL}/click",
                headers=headers,
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Click analysis: {result['result']}")
                return result
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

def main():
    """Demo the HTTP API"""
    print("🧪 UI-TARS HTTP API Client Demo")
    print("=" * 40)
    
    # Check if server is running
    if not test_health():
        print("❌ Server is not running. Start it with:")
        print("   python3 northstar_mcp.py")
        return
    
    # Test image
    test_image = "test_image.png"
    if not Path(test_image).exists():
        print(f"❌ Test image {test_image} not found")
        return
    
    print(f"\n🔑 Using API key: {API_KEY}")
    print(f"📷 Test image: {test_image}")
    
    print("\n1️⃣ Testing Image Analysis...")
    analyze_image(test_image, "What website is shown in this image?")
    
    print("\n2️⃣ Testing Image Description...")
    describe_image(test_image)
    
    print("\n3️⃣ Testing Click Analysis...")
    click_analysis(test_image, 300, 150)
    
    print("\n✅ Demo completed!")
    print("\n💡 Curl examples:")
    print(f"curl -X POST '{SERVER_URL}/analyze' \\")
    print(f"     -H 'Authorization: Bearer {API_KEY}' \\")
    print(f"     -F 'file=@{test_image}' \\")
    print(f"     -F 'query=What do you see?'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    main()
