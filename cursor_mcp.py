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
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from PIL import Image
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-screen-client")

# Configuration - reads from environment for remote connections
GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "http://localhost:8000")
GPU_API_KEY = os.getenv("GPU_API_KEY", "ui-tars-L_PJlAW2_0j4uU_hcPYGhxlhTQOlAFGLayNHwpLMPMw")

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

    def capture_screenshot(self) -> Tuple[Image.Image, Tuple[int, int]]:
        """Capture screenshot and return both original and resized versions"""
        # Capture full screen
        screenshot = pyautogui.screenshot()
        original_size = screenshot.size
        
        logger.info(f"Captured screenshot: {original_size}")
        
        # Resize to target size for GPU server while maintaining aspect ratio
        resized = screenshot.copy()
        resized.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        
        # Create final image with white background
        final_image = Image.new("RGB", self.target_size, (255, 255, 255))
        offset = ((self.target_size[0] - resized.size[0]) // 2, 
                 (self.target_size[1] - resized.size[1]) // 2)
        final_image.paste(resized, offset)
        
        return final_image, original_size

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

    async def send_to_gpu_server(self, image: Image.Image, query: str) -> dict:
        """Send image to GPU server for analysis"""
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name, 'PNG')
            tmp_file_path = tmp_file.name
        
        try:
            headers = {"Authorization": f"Bearer {GPU_API_KEY}"}
            
            with open(tmp_file_path, 'rb') as f:
                files = {"file": f}
                data = {
                    "query": query,
                    "max_tokens": 500
                }
                
                logger.info(f"Sending request to GPU server: {query}")
                
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
                    
        finally:
            # Clean up temp file
            try:
                import os
                os.unlink(tmp_file_path)
            except:
                pass

    async def _click_element(self, arguments: dict) -> list[types.TextContent]:
        """Take screenshot, find element, and click it"""
        element_description = arguments.get("element_description", "")
        
        if not element_description:
            raise ValueError("element_description is required")
        
        # Capture screenshot
        image, original_size = self.capture_screenshot()
        
        # Create query for finding clickable element
        query = f"I want to click on the {element_description}. Please provide the exact coordinates where I should click. Respond with coordinates in format (x, y)."
        
        # Send to GPU server
        result = await self.send_to_gpu_server(image, query)
        
        # Extract coordinates
        if result.get('coordinates'):
            # Scale coordinates back to original resolution
            scaled_x, scaled_y = self.scale_coordinates(result['coordinates'], original_size)
            
            # Perform click
            pyautogui.click(scaled_x, scaled_y)
            
            return [types.TextContent(
                type="text",
                text=f"Successfully clicked {element_description} at coordinates ({scaled_x}, {scaled_y}). Analysis: {result['result']}"
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
        image, original_size = self.capture_screenshot()
        
        # Send to GPU server
        result = await self.send_to_gpu_server(image, query)
        
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
        image, original_size = self.capture_screenshot()
        
        # Create query for finding element coordinates
        query = f"Where is the {element_description} located? Please provide the exact coordinates in format (x, y)."
        
        # Send to GPU server
        result = await self.send_to_gpu_server(image, query)
        
        # Extract and scale coordinates
        if result.get('coordinates'):
            scaled_x, scaled_y = self.scale_coordinates(result['coordinates'], original_size)
            
            return [types.TextContent(
                type="text",
                text=f"Found {element_description} at coordinates ({scaled_x}, {scaled_y}). Analysis: {result['result']}"
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
    print("🖥️  Starting Screen Automation MCP Client")
    print(f"🎯 GPU Server: {GPU_SERVER_URL}")
    print("🔧 Available tools:")
    print("  - click_element: Find and click UI elements")
    print("  - analyze_screen: Analyze what's on screen")
    print("  - find_coordinates: Get coordinates of elements")
    
    client = ScreenAutomationClient()
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Client error: {e}")

if __name__ == "__main__":
    main()
