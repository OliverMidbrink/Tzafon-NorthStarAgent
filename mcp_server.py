#!/usr/bin/env python3
"""
MCP Server for UI-TARS-2B Vision Language Model

This server provides vision-language capabilities through the Model Context Protocol (MCP).
It can analyze images and respond to queries about them.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import our UI TARS functionality
from ui_tars_2b_infer import run_inference, DEFAULT_MODEL_ID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ui-tars-mcp-server")

# Global variables for model caching
MODEL_CACHE = {}
CACHED_MODEL_ID = None

class UITARSMCPServer:
    def __init__(self):
        self.server = Server("ui-tars-vision")
        self.model_id = DEFAULT_MODEL_ID
        self.max_tokens = 3000
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self):
        """Register MCP tools for vision analysis"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools"""
            return [
                types.Tool(
                    name="analyze_image",
                    description="Analyze an image and answer questions about it using UI-TARS-2B vision-language model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Path to the image file to analyze"
                            },
                            "query": {
                                "type": "string",
                                "description": "Question or instruction about the image"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum number of tokens to generate (default: 3000)",
                                "default": 3000
                            }
                        },
                        "required": ["image_path", "query"]
                    },
                ),
                types.Tool(
                    name="describe_image",
                    description="Get a detailed description of an image",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Path to the image file to describe"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum number of tokens to generate (default: 3000)",
                                "default": 3000
                            }
                        },
                        "required": ["image_path"]
                    },
                ),
                types.Tool(
                    name="click_coordinate",
                    description="Analyze what would happen if clicking at specific coordinates in an image",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Path to the image file"
                            },
                            "x": {
                                "type": "integer",
                                "description": "X coordinate to click"
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate to click"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum number of tokens to generate (default: 3000)",
                                "default": 3000
                            }
                        },
                        "required": ["image_path", "x", "y"]
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent]:
            """Handle tool calls"""
            
            if not arguments:
                raise ValueError("No arguments provided")
                
            try:
                if name == "analyze_image":
                    return await self._analyze_image(arguments)
                elif name == "describe_image":
                    return await self._describe_image(arguments)
                elif name == "click_coordinate":
                    return await self._click_coordinate(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

    async def _analyze_image(self, arguments: dict) -> list[types.TextContent]:
        """Analyze an image with a custom query"""
        image_path = arguments.get("image_path")
        query = arguments.get("query")
        max_tokens = arguments.get("max_tokens", self.max_tokens)
        
        if not image_path or not query:
            raise ValueError("Both image_path and query are required")
            
        # Validate image exists
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
            
        logger.info(f"Analyzing image: {image_path} with query: {query}")
        
        # Run inference
        result = await asyncio.get_event_loop().run_in_executor(
            None, 
            run_inference,
            self.model_id,
            image_path,
            query,
            max_tokens
        )
        
        return [types.TextContent(
            type="text",
            text=result
        )]

    async def _describe_image(self, arguments: dict) -> list[types.TextContent]:
        """Get a detailed description of an image"""
        image_path = arguments.get("image_path")
        max_tokens = arguments.get("max_tokens", self.max_tokens)
        
        if not image_path:
            raise ValueError("image_path is required")
            
        # Validate image exists
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
            
        logger.info(f"Describing image: {image_path}")
        
        query = "Describe this image in detail, including all visible elements, text, UI components, and their layout."
        
        # Run inference
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_inference,
            self.model_id,
            image_path,
            query,
            max_tokens
        )
        
        return [types.TextContent(
            type="text",
            text=result
        )]

    async def _click_coordinate(self, arguments: dict) -> list[types.TextContent]:
        """Analyze what would happen when clicking at specific coordinates"""
        image_path = arguments.get("image_path")
        x = arguments.get("x")
        y = arguments.get("y")
        max_tokens = arguments.get("max_tokens", self.max_tokens)
        
        if not image_path or x is None or y is None:
            raise ValueError("image_path, x, and y coordinates are required")
            
        # Validate image exists
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
            
        logger.info(f"Analyzing click at ({x}, {y}) in image: {image_path}")
        
        query = f"What would happen if I clicked at coordinates ({x}, {y}) in this image? Describe the element at that location and the expected action or result."
        
        # Run inference
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_inference,
            self.model_id,
            image_path,
            query,
            max_tokens
        )
        
        return [types.TextContent(
            type="text",
            text=result
        )]

    async def run(self):
        """Run the MCP server"""
        # MCP server with stdio transport
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ui-tars-vision",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="UI-TARS MCP Server")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID to use"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Default maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Create and run server
    server = UITARSMCPServer()
    server.model_id = args.model_id
    server.max_tokens = args.max_tokens
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
