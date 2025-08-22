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
import json as _json
import re as _re
from datetime import datetime
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from PIL import Image
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Import our UI TARS functionality
from ui_tars_model import run_inference, DEFAULT_MODEL_ID

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ui-tars-http-server")

# Load API key from environment, file, or generate new one
def load_api_key():
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
    
    # Finally generate a new one
    api_key = "ui-tars-" + secrets.token_urlsafe(32)
    return api_key, "generated"

API_KEY, key_source = load_api_key()
print(f"🔑 API Key: {API_KEY} (from {key_source})")

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

class FileAnalysisRequest(BaseModel):
    filename: str
    query: str
    max_tokens: Optional[int] = 3000

class AnalysisResponse(BaseModel):
    result: str
    coordinates: Optional[dict] = None
    image_size: Optional[str] = None

def extract_coordinates(text: str) -> Optional[dict]:
    """Extract coordinates from model output (prefers JSON)."""
    if not text:
        return None

    # Try fenced code blocks first
    fenced = _re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, _re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text.strip()

    # Try full JSON
    try:
        parsed = _json.loads(candidate)
        if isinstance(parsed, dict) and "x" in parsed and "y" in parsed:
            return {"x": int(parsed["x"]), "y": int(parsed["y"])}
    except Exception:
        pass

    # Try to find a JSON object substring
    obj_match = _re.search(r"\{[^{}]*\}", candidate, _re.DOTALL)
    if obj_match:
        try:
            parsed = _json.loads(obj_match.group(0))
            if isinstance(parsed, dict) and "x" in parsed and "y" in parsed:
                return {"x": int(parsed["x"]), "y": int(parsed["y"])}
        except Exception:
            pass

    # Fallback: classic patterns like (x, y) etc.
    patterns = [
        r'\((\d+),\s*(\d+)\)',
        r'<click>(\d+),\s*(\d+)</click>', 
        r'coordinates?\s*\((\d+),\s*(\d+)\)',
        r'click.*?(\d+),\s*(\d+)',
        r'at\s+(\d+),\s*(\d+)'
    ]
    for pattern in patterns:
        match = _re.search(pattern, text, _re.IGNORECASE)
        if match:
            try:
                x, y = int(match.group(1)), int(match.group(2))
                if 0 <= x <= 20000 and 0 <= y <= 20000:
                    return {"x": x, "y": y}
            except Exception:
                continue

    return None

def scale_coords_to_original(original_w: int, original_h: int, resized_w: int, resized_h: int, model_x: int, model_y: int) -> dict:
    """Map coordinates from smart-resized image back to the original image using exact scale factors."""
    if resized_w <= 0 or resized_h <= 0:
        return {"x": 0, "y": 0}
    scale_x = original_w / resized_w
    scale_y = original_h / resized_h
    final_x = int(round(model_x * scale_x))
    final_y = int(round(model_y * scale_y))
    final_x = max(0, min(final_x, max(0, original_w - 1)))
    final_y = max(0, min(final_y, max(0, original_h - 1)))
    return {"x": final_x, "y": final_y}

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")

def save_model_input_image(src_path: str) -> str:
    """Save the proportionally downscaled (major axis 1024, no padding) image
    that mirrors the model input into model_inputted_images_log/.

    Returns the saved path or empty string on failure.
    """
    try:
        out_dir = "model_inputted_images_log"
        os.makedirs(out_dir, exist_ok=True)
        with Image.open(src_path).convert("RGB") as img:
            down = img.copy()
            down.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            name = f"model_input_{ts}_{uuid.uuid4().hex}.png"
            out_path = os.path.join(out_dir, name)
            down.save(out_path, "PNG")
            logger.info(f"Saved model input image: {out_path} ({down.size[0]}x{down.size[1]})")
            return out_path
    except Exception as e:
        logger.warning(f"Failed saving model input image: {e}")
        return ""

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
    scene: str = Form("computer"),
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
        # Save the exact model input size (we still keep a debug copy proportional 1024 canvas separately)
        _ = save_model_input_image(tmp_file_path)
        # Persist a 1024x1024 low-scale canvas for inspection
        try:
            os.makedirs("tmp", exist_ok=True)
            with Image.open(tmp_file_path).convert("RGB") as img_dbg:
                dbg = img_dbg.copy()
                dbg.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (1024, 1024), (255, 255, 255))
                off = ((1024 - dbg.size[0]) // 2, (1024 - dbg.size[1]) // 2)
                canvas.paste(dbg, off)
                stem = Path(tmp_file_path).stem
                out = os.path.join("tmp", f"low_scale_{stem}.png")
                canvas.save(out, "PNG")
                logger.info(f"Saved low-scale image: {out}")
        except Exception as e:
            logger.warning(f"Low-scale save failed: {e}")
        
        # Run inference
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_inference,
            DEFAULT_MODEL_ID,
            tmp_file_path,
            query,
            max_tokens,
            scene
        )
        
        # Extract coordinates if present and convert back to original size (using smart-resized dimensions)
        coordinates = extract_coordinates(result)
        if coordinates is not None:
            try:
                with Image.open(tmp_file_path) as img:
                    original_w, original_h = img.size
                # Recompute smart-resized dimensions to match ui_tars_model
                from ui_tars_model import smart_resize, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS
                resized_h, resized_w = smart_resize(original_h, original_w, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
                scaled = scale_coords_to_original(original_w, original_h, resized_w, resized_h, coordinates["x"], coordinates["y"])
                coordinates = scaled
            except Exception as e:
                logger.error(f"Failed to back-scale coordinates: {e}")
        
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

@app.post("/analyze_file", response_model=AnalysisResponse)
async def analyze_file(
    request: FileAnalysisRequest,
    api_key: str = Depends(verify_api_key)
):
    """Analyze an image file from tmp folder"""
    
    filepath = os.path.join("tmp", request.filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    # Check if GPU is busy
    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="GPU busy, try again later")
    
    try:
        logger.info(f"Analyzing file {request.filename} with query: {request.query}")
        
        # Run inference
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_inference,
            DEFAULT_MODEL_ID,
            filepath,
            request.query,
            request.max_tokens,
            getattr(request, "scene", "computer")
        )
        
        # Extract coordinates if present
        coordinates = extract_coordinates(result)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        return AnalysisResponse(
            result=result,
            coordinates=coordinates,
            image_size=f"{file_size} bytes"
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Release lock but keep temp file for potential reuse
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
        # Save the exact model input size (debug helper retained)
        _ = save_model_input_image(tmp_file_path)
        # Persist a 1024x1024 low-scale canvas for inspection
        try:
            os.makedirs("tmp", exist_ok=True)
            with Image.open(tmp_file_path).convert("RGB") as img_dbg:
                dbg = img_dbg.copy()
                dbg.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (1024, 1024), (255, 255, 255))
                off = ((1024 - dbg.size[0]) // 2, (1024 - dbg.size[1]) // 2)
                canvas.paste(dbg, off)
                stem = Path(tmp_file_path).stem
                out = os.path.join("tmp", f"low_scale_{stem}.png")
                canvas.save(out, "PNG")
                logger.info(f"Saved low-scale image: {out}")
        except Exception as e:
            logger.warning(f"Low-scale save failed: {e}")
        
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
        if coordinates is not None:
            try:
                with Image.open(tmp_file_path) as img:
                    original_w, original_h = img.size
                if original_w >= original_h:
                    down_w = 1024
                    down_h = int(round(original_h * (1024 / original_w)))
                else:
                    down_h = 1024
                    down_w = int(round(original_w * (1024 / original_h)))
                coordinates = scale_coords_to_original(original_w, original_h, down_w, down_h, coordinates["x"], coordinates["y"])
            except Exception as e:
                logger.error(f"Failed to back-scale coordinates: {e}")
        
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
        # Save the exact downscaled image used as model input (proportional, no padding)
        _ = save_model_input_image(tmp_file_path)
        
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
    print("🌟 NorthStar MCP Server - UI-TARS Vision API")
    print("=" * 50)
    print(f"🔑 API Key: {API_KEY} (from {key_source})")
    print("🎯 GPU Model: UI-TARS-2B-SFT")
    print("🔒 Single-threaded GPU processing")
    print("")
    print("📡 Available endpoints:")
    print("  POST /analyze - Custom image analysis with coordinates")
    print("  POST /describe - Detailed image descriptions") 
    print("  POST /click - Click coordinate analysis")
    print("  GET /health - Server and GPU health check")
    print("")
    print("🔐 Authentication: Bearer token required")
    print("💡 Example usage:")
    print(f'  curl -X POST "http://localhost:8000/analyze" \\')
    print(f'       -H "Authorization: Bearer {API_KEY}" \\')
    print(f'       -F "file=@screenshot.png" \\')
    print(f'       -F "query=Where is the close button?"')
    print("")
    print("🚀 Starting server on http://0.0.0.0:8000...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
