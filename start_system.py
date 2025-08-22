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
    """Update the API key in the MCP client"""
    with open("mcp_client.py", "r") as f:
        content = f.read()
    
    # Find and replace the API key line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'GPU_API_KEY = ' in line and 'ui-tars-' in line:
            lines[i] = f'GPU_API_KEY = "{api_key}"  # Auto-updated'
            break
    
    with open("mcp_client.py", "w") as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Updated mcp_client.py with API key")

def main():
    print("🚀 Starting Screen Automation System")
    print("=" * 50)
    
    # Generate or use existing API key
    api_key = os.getenv("MCP_API_KEY")
    if not api_key:
        api_key = generate_api_key()
        print(f"🔑 Generated API key: {api_key}")
    else:
        print(f"🔑 Using API key: {api_key}")
    
    # Update client with API key
    update_client_api_key(api_key)
    
    print("\n📋 System Components:")
    print("1. GPU Server (mcp_http_server.py) - Runs UI-TARS model")
    print("2. MCP Client (mcp_client.py) - Screen automation for Cursor")
    
    print(f"\n🔧 To start GPU server:")
    print(f"   MCP_API_KEY={api_key} python3 mcp_http_server.py")
    
    print(f"\n🔧 To start MCP client:")
    print(f"   python3 mcp_client.py")
    
    print(f"\n💡 For Cursor MCP config:")
    print(f'   "screen-automation": {{')
    print(f'     "command": "python3",')
    print(f'     "args": ["{os.path.abspath("mcp_client.py")}"],')
    print(f'     "cwd": "{os.getcwd()}"')
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
                sys.executable, "mcp_http_server.py"
            ], env=env)
        except KeyboardInterrupt:
            print("\n⏹️  GPU server stopped")

if __name__ == "__main__":
    main()
