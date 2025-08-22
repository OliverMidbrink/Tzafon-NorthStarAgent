#!/usr/bin/env python3
"""
HTTP MCP Server for UI-TARS-2B Vision Language Model

This server provides vision-language capabilities over HTTP with API key authentication.
"""

import asyncio
import json
import logging
import os
import re
import secrets
import tempfile
import threading
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Import our UI TARS functionality
from ui_tars_2b_infer import run_inference, DEFAULT_MODEL_ID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ui-tars-http-server")

# Generate or load API key
API_KEY = os.getenv("MCP_API_KEY", "ui-tars-" + secrets.token_urlsafe(32))
print(f"🔑 API Key: {API_KEY}")

# Single request lock for GPU
gpu_lock = threading.Lock()

# FastAPI app
app = FastAPI(
    title="UI-TARS Vision API",
    description="Vision-language analysis using UI-TARS-2B model",
    version="0.1.0"
)

# Security
security = HTTPBearer()

class AnalysisRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 3000

class AnalysisResponse(BaseModel):
    result: str
    coordinates: Optional[dict] = None
    image_size: Optional[str] = None

def extract_coordinates(text: str) -> Optional[dict]:
    """Extract coordinates from model output"""
    # Look for patterns like (x, y) or <click>x, y</click> or coordinates (x, y)
    patterns = [
        r'\((\d+),\s*(\d+)\)',
        r'<click>(\d+),\s*(\d+)</click>', 
        r'coordinates?\s*\((\d+),\s*(\d+)\)',
        r'click.*?(\d+),\s*(\d+)',
        r'at\s+(\d+),\s*(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            # Basic sanity check for coordinates
            if 0 <= x <= 5000 and 0 <= y <= 5000:
                return {"x": x, "y": y}
    
    return None

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "UI-TARS Vision API",
        "version": "0.1.0",
        "endpoints": {
            "/analyze": "POST - Analyze image with custom query",
            "/describe": "POST - Get detailed image description", 
            "/click": "POST - Analyze what happens when clicking coordinates"
        },
        "auth": "Bearer token required"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        
        return {
            "status": "healthy",
            "cuda_available": cuda_available,
            "device": device_name,
            "model": DEFAULT_MODEL_ID
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e)
        }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    query: str = Form(...),
    max_tokens: int = Form(3000),
    api_key: str = Depends(verify_api_key)
):
    """Analyze an image with a custom query"""
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check if GPU is busy
    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="GPU busy, try again later")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        logger.info(f"Analyzing image with query: {query}")
        
        # Run inference
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_inference,
            DEFAULT_MODEL_ID,
            tmp_file_path,
            query,
            max_tokens
        )
        
        # Extract coordinates if present
        coordinates = extract_coordinates(result)
        
        return AnalysisResponse(
            result=result,
            coordinates=coordinates,
            image_size=f"{content.__len__()} bytes"
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temp file and release lock
        cleanup_temp_file(tmp_file_path)
        gpu_lock.release()

@app.post("/describe", response_model=AnalysisResponse)
async def describe_image(
    file: UploadFile = File(...),
    max_tokens: int = Form(3000),
    api_key: str = Depends(verify_api_key)
):
    """Get a detailed description of an image"""
    
    query = "Describe this image in detail, including all visible elements, text, UI components, and their layout."
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check if GPU is busy
    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="GPU busy, try again later")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        logger.info(f"Describing image")
        
        # Run inference
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_inference,
            DEFAULT_MODEL_ID,
            tmp_file_path,
            query,
            max_tokens
        )
        
        coordinates = extract_coordinates(result)
        
        return AnalysisResponse(
            result=result,
            coordinates=coordinates,
            image_size=f"{content.__len__()} bytes"
        )
        
    except Exception as e:
        logger.error(f"Description failed: {e}")
        raise HTTPException(status_code=500, detail=f"Description failed: {str(e)}")
    
    finally:
        # Clean up temp file and release lock
        cleanup_temp_file(tmp_file_path)
        gpu_lock.release()

@app.post("/click", response_model=AnalysisResponse)
async def click_coordinate(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    max_tokens: int = Form(3000),
    api_key: str = Depends(verify_api_key)
):
    """Analyze what would happen when clicking at specific coordinates"""
    
    query = f"What would happen if I clicked at coordinates ({x}, {y}) in this image? Describe the element at that location and the expected action or result."
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        logger.info(f"Analyzing click at ({x}, {y})")
        
        # Run inference
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_inference,
            DEFAULT_MODEL_ID,
            tmp_file_path,
            query,
            max_tokens
        )
        
        return AnalysisResponse(
            result=result,
            image_size=f"{content.__len__()} bytes"
        )
        
    except Exception as e:
        logger.error(f"Click analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Click analysis failed: {str(e)}")
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_file_path)
        except:
            pass

def main():
    """Main entry point"""
    print("🚀 Starting UI-TARS HTTP Server")
    print(f"🔑 API Key: {API_KEY}")
    print("📡 Endpoints:")
    print("  POST /analyze - Analyze image with query")
    print("  POST /describe - Describe image") 
    print("  POST /click - Analyze click coordinates")
    print("  GET /health - Health check")
    print("")
    print("🔐 Authentication: Bearer token required")
    print("💡 Example curl:")
    print(f'  curl -X POST "http://localhost:8000/analyze" \\')
    print(f'       -H "Authorization: Bearer {API_KEY}" \\')
    print(f'       -F "file=@image.png" \\')
    print(f'       -F "query=What do you see?"')
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
