#!/usr/bin/env python3
"""
Startup script for the complete screen automation system
"""

import os
import secrets
import subprocess
import sys
import time

def generate_api_key():
    """Generate a consistent API key"""
    return "ui-tars-" + secrets.token_urlsafe(32)

def update_client_api_key(api_key: str):
    """Update the API key in the Cursor MCP client"""
    with open("cursor_mcp.py", "r") as f:
        content = f.read()
    
    # Find and replace the API key line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'GPU_API_KEY = ' in line and 'ui-tars-' in line:
            lines[i] = f'GPU_API_KEY = "{api_key}"  # Auto-updated'
            break
    
    with open("cursor_mcp.py", "w") as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Updated cursor_mcp.py with API key")

def load_api_key():
    """Load API key from environment, file, or generate new one"""
    # First try environment variable
    api_key = os.getenv("MCP_API_KEY")
    if api_key:
        return api_key, "environment"
    
    # Then try .api_key.txt file
    try:
        with open(".api_key.txt", "r") as f:
            api_key = f.read().strip()
            if api_key:
                return api_key, "file"
    except FileNotFoundError:
        pass
    
    # Finally generate a new one and save it
    api_key = generate_api_key()
    try:
        with open(".api_key.txt", "w") as f:
            f.write(api_key)
        return api_key, "generated (saved to .api_key.txt)"
    except:
        return api_key, "generated"

def main():
    print("🚀 Starting Screen Automation System")
    print("=" * 50)
    
    # Load API key from various sources
    api_key, key_source = load_api_key()
    print(f"🔑 API Key: {api_key} (from {key_source})")
    
    # Update client with API key
    update_client_api_key(api_key)
    
    print("\n📋 System Components:")
    print("1. NorthStar MCP (northstar_mcp.py) - GPU server running UI-TARS model")
    print("2. Cursor MCP (cursor_mcp.py) - Client for screen automation in Cursor")
    
    print(f"\n🔧 To start NorthStar GPU server:")
    print(f"   MCP_API_KEY={api_key} python3 northstar_mcp.py")
    
    print(f"\n🔧 To start Cursor MCP client:")
    print(f"   python3 cursor_mcp.py")
    
    print(f"\n💡 For Cursor MCP config (on MacBook):")
    print(f'   "screen-automation": {{')
    print(f'     "command": "python3",')
    print(f'     "args": ["/path/to/cursor_mcp.py"],')
    print(f'     "cwd": "/path/to/client/directory",')
    print(f'     "env": {{')
    print(f'       "GPU_SERVER_URL": "http://YOUR_STATIC_IP:8000",')
    print(f'       "GPU_API_KEY": "{api_key}"')
    print(f'     }}')
    print(f'   }}')
    
    print(f"\n🎯 Available MCP tools in Cursor:")
    print("   - click_element: 'Click the X button'")
    print("   - analyze_screen: 'What is on the screen?'")
    print("   - find_coordinates: 'Where is the submit button?'")
    
    # Ask if user wants to start components
    print(f"\n" + "=" * 50)
    choice = input("Start GPU server now? (y/n): ").lower().strip()
    
    if choice == 'y':
        print("🚀 Starting GPU server...")
        env = os.environ.copy()
        env["MCP_API_KEY"] = api_key
        
        try:
            subprocess.run([
                sys.executable, "northstar_mcp.py"
            ], env=env)
        except KeyboardInterrupt:
            print("\n⏹️  NorthStar GPU server stopped")

if __name__ == "__main__":
    main()
