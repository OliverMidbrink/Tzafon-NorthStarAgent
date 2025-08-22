#!/usr/bin/env python3
"""
MCP Client for Screen Automation

This acts as an MCP server for Cursor/Claude but sends requests to the GPU server.
Handles screenshot capture, coordinate scaling, and click automation.
"""

import asyncio
import io
import logging
import mcp.server.stdio
import mcp.types as types
import os
import pyautogui
import requests
import tempfile
import uuid
import time
import base64
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-screen-client")

# Configuration - reads from environment, file, or fallback
def load_config():
    # Server URL
    server_url = os.getenv("GPU_SERVER_URL", "http://localhost:8000")
    
    # API Key priority: environment > file > fallback
    api_key = os.getenv("GPU_API_KEY")
    if not api_key:
        try:
            # Try current directory first, then script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            api_key_paths = [".api_key.txt", os.path.join(script_dir, ".api_key.txt")]
            
            for path in api_key_paths:
                try:
                    with open(path, "r") as f:
                        api_key = f.read().strip()
                        break
                except FileNotFoundError:
                    continue
            else:
                api_key = "ui-tars-default-key"
        except Exception:
            api_key = "ui-tars-default-key"
    
    return server_url, api_key

GPU_SERVER_URL, GPU_API_KEY = load_config()
logger.info(f"Loaded API key: {GPU_API_KEY[:20]}... from config")

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

class ScreenAutomationClient:
    def __init__(self):
        self.server = Server("screen-automation")
        self.target_size = (1024, 1024)  # Size sent to GPU server
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools for screen automation"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools"""
            return [
                types.Tool(
                    name="click_element",
                    description="Take a screenshot, find and click on a UI element based on description",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_description": {
                                "type": "string",
                                "description": "Description of the element to click (e.g., 'X button', 'submit button', 'close icon')"
                            }
                        },
                        "required": ["element_description"]
                    },
                ),
                types.Tool(
                    name="analyze_screen",
                    description="Take a screenshot and analyze what's visible on screen",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to analyze about the screen (optional, defaults to general description)"
                            }
                        },
                        "required": []
                    },
                ),
                types.Tool(
                    name="find_coordinates",
                    description="Take a screenshot and find coordinates of a specific element",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "element_description": {
                                "type": "string",
                                "description": "Description of the element to locate"
                            }
                        },
                        "required": ["element_description"]
                    },
                ),
                
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent]:
            """Handle tool calls"""
            
            if not arguments:
                arguments = {}
                
            try:
                if name == "click_element":
                    return await self._click_element(arguments)
                elif name == "analyze_screen":
                    return await self._analyze_screen(arguments)
                elif name == "find_coordinates":
                    return await self._find_coordinates(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

    def capture_screenshot(self) -> Tuple[str, Tuple[int, int]]:
        """Capture screenshot (original size), save to absolute tmp folder, return filename and original size"""
        # Capture full screen at original resolution
        screenshot = pyautogui.screenshot()
        original_size = screenshot.size
        logger.info(f"Captured screenshot: {original_size}")

        # Ensure absolute tmp directory next to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(script_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        # Save original image (no resizing) with UUID name
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(tmp_dir, filename)
        screenshot.save(filepath, "PNG")

        # Also save a proportional low-scale copy (major axis 1024, no padding)
        resized = screenshot.copy()
        resized.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        low_path = os.path.join(tmp_dir, f"low_scale_{Path(filename).stem}.png")
        resized.save(low_path, "PNG")
        logger.info(f"Saved proportional low-scale client image: {low_path}")

        return filename, original_size

    def scale_coordinates(self, coords: dict, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """Scale coordinates from 1024x1024 back to original screen resolution"""
        if not coords or 'x' not in coords or 'y' not in coords:
            raise ValueError("Invalid coordinates")
        
        # Get the actual displayed size (thumbnail size)
        original_w, original_h = original_size
        
        # Calculate scaling factor (same as used in thumbnail)
        scale_x = original_w / self.target_size[0]
        scale_y = original_h / self.target_size[1]
        scale = min(scale_x, scale_y)  # thumbnail uses min
        
        # Calculate offset (centering)
        displayed_w = int(original_w / scale)
        displayed_h = int(original_h / scale)
        offset_x = (self.target_size[0] - displayed_w) // 2
        offset_y = (self.target_size[1] - displayed_h) // 2
        
        # Convert coordinates
        adj_x = coords['x'] - offset_x
        adj_y = coords['y'] - offset_y
        
        # Scale back to original
        final_x = int(adj_x * scale)
        final_y = int(adj_y * scale)
        
        logger.info(f"Scaled coords ({coords['x']}, {coords['y']}) -> ({final_x}, {final_y})")
        
        return final_x, final_y

    async def send_to_gpu_server(self, filename: str, query: str, scene: str = "computer") -> dict:
        """Send image file to GPU server for analysis with a scene hint (computer|grounding)."""
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "tmp", filename)
        
        try:
            headers = {"Authorization": f"Bearer {GPU_API_KEY}"}
            
            with open(filepath, 'rb') as f:
                files = {"file": (filename, f, "image/png")}
                data = {
                    "query": query,
                    "max_tokens": 500,
                    "scene": scene,
                }
                
                logger.info(f"Sending request to GPU server: {query}")
                logger.info(f"Image filename: {filename}")
                
                response = requests.post(
                    f"{GPU_SERVER_URL}/analyze",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30
                )
                
            if response.status_code == 200:
                result = response.json()
                logger.info(f"GPU server response: {result}")
                return result
            elif response.status_code == 503:
                raise Exception("GPU server is busy, try again later")
            else:
                raise Exception(f"GPU server error {response.status_code}: {response.text}")
                    
        except Exception as e:
            logger.error(f"Error sending to GPU server: {e}")
            raise


    async def _click_element(self, arguments: dict) -> list[types.TextContent]:
        """Take screenshot, find element, and click it"""
        element_description = arguments.get("element_description", "")
        
        if not element_description:
            raise ValueError("element_description is required")
        
        # Capture screenshot
        filename, original_size = self.capture_screenshot()
        
        # Create concise instruction (server will wrap with grounding prompt)
        query = f"Click the {element_description}"
        
        # Send to GPU server
        result = await self.send_to_gpu_server(filename, query, scene="grounding")
        
        # Extract coordinates
        if result.get('coordinates'):
            # Server returns coordinates already in original resolution
            coords = result['coordinates']
            target_x, target_y = coords['x'], coords['y']
            # Smoothly move for 1s so it is visible, then mouseDown/Up
            pyautogui.moveTo(target_x, target_y, duration=1.0)
            pyautogui.mouseDown()
            time.sleep(0.05)
            pyautogui.mouseUp()
            return [types.TextContent(
                type="text",
                text=f"Successfully clicked {element_description} at coordinates ({target_x}, {target_y}). Analysis: {result['result']}"
            )]
        else:
            return [types.TextContent(
                type="text",
                text=f"Could not find coordinates for {element_description}. Analysis: {result['result']}"
            )]

    async def _analyze_screen(self, arguments: dict) -> list[types.TextContent]:
        """Take screenshot and analyze what's visible"""
        query = arguments.get("query", "Describe what is visible on this screen in detail.")
        
        # Capture screenshot
        filename, original_size = self.capture_screenshot()
        
        # Send to GPU server
        result = await self.send_to_gpu_server(filename, query, scene="computer")
        
        return [types.TextContent(
            type="text",
            text=f"Screen analysis: {result['result']}"
        )]

    async def _find_coordinates(self, arguments: dict) -> list[types.TextContent]:
        """Find coordinates of an element without clicking"""
        element_description = arguments.get("element_description", "")
        
        if not element_description:
            raise ValueError("element_description is required")
        
        # Capture screenshot
        filename, original_size = self.capture_screenshot()
        
        # Create concise instruction (server will wrap with grounding prompt)
        query = f"Find the {element_description}"
        
        # Send to GPU server
        result = await self.send_to_gpu_server(filename, query, scene="grounding")
        
        # Extract coordinates (already original scale from server)
        if result.get('coordinates'):
            coords = result['coordinates']
            return [types.TextContent(
                type="text",
                text=f"Found {element_description} at coordinates ({coords['x']}, {coords['y']}). Analysis: {result['result']}"
            )]
        else:
            return [types.TextContent(
                type="text",
                text=f"Could not find coordinates for {element_description}. Analysis: {result['result']}"
            )]

    

    async def run(self):
        """Run the MCP client server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="screen-automation",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

def main():
    """Main entry point"""
    print("🖥️  Cursor MCP Client - Screen Automation")
    print("=" * 50)
    print(f"🎯 NorthStar Server: {GPU_SERVER_URL}")
    print(f"🔑 API Key: {GPU_API_KEY[:20]}...")
    print("")
    print("🔧 Available MCP tools for Cursor:")
    print("  - click_element: 'Click the X button'")
    print("  - analyze_screen: 'What is on the screen?'")
    print("  - find_coordinates: 'Where is the login button?'")
    print("")
    print("📱 This client:")
    print("  • Captures screenshots locally")
    print("  • Sends to NorthStar for AI analysis")  
    print("  • Performs mouse clicks locally")
    print("  • Scales coordinates automatically")
    print("")
    print("🚀 Starting MCP server for Cursor...")
    
    client = ScreenAutomationClient()
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Client error: {e}")

if __name__ == "__main__":
    main()
