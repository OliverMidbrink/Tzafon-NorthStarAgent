# NorthStar Agent - UI-TARS Vision System

**GPU-accelerated screen automation using UI-TARS-2B vision model**

## 🚀 Quick Start

**Server (CUDA/GPU):**
```bash
source .venv/bin/activate
python3 northstar_mcp.py
```

**Client (MacBook/Cursor):**
```bash
python3 cursor_mcp.py
```

## 🎯 Components

- **`northstar_mcp.py`** - GPU server running UI-TARS model
- **`cursor_mcp.py`** - MCP client for Cursor screen automation
- **`.api_key.txt`** - Shared API key (auto-generated)

## 💡 Usage

In Cursor: *"Click the close button"* → Screenshots → AI analysis → Click automation
