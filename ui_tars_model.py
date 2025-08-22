import argparse
import os
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForVision2Seq

# UI-TARS image scaling constants
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


DEFAULT_MODEL_ID = "ByteDance-Seed/UI-TARS-2B-SFT"

# Prompt templates adapted from UI-TARS repo README/prompt files
# Computer-use agent prompt (concise version)
PROMPT_COMPUTER = (
    "You are a GUI agent. You are given a task and your action history, with screenshots. "
    "You need to perform the next action to complete the task.\n\n"
    "## Output Format\n"
    "Thought: ...\n"
    "Action: ...\n\n"
    "## Action Space\n"
    "click(start_box='\u003c|box_start|\u003e(x1,y1)\u003c|box_end|\u003e')\n"
    "left_double(start_box='\u003c|box_start|\u003e(x1,y1)\u003c|box_end|\u003e')\n"
    "right_single(start_box='\u003c|box_start|\u003e(x1,y1)\u003c|box_end|\u003e')\n"
    "drag(start_box='\u003c|box_start|\u003e(x1,y1)\u003c|box_end|\u003e', end_box='\u003c|box_start|\u003e(x3,y3)\u003c|box_end|\u003e')\n"
    "hotkey(key='')\n"
    "type(content='')\n"
    "scroll(start_box='\u003c|box_start|\u003e(x1,y1)\u003c|box_end|\u003e', direction='down or up or right or left')\n"
    "wait()\n"
    "finished()\n\n"
    "## Note\n"
    "- Summarize your next action (with its target element) in one sentence in `Thought`.\n"
    "\n\n## User Instruction\n"
)

# Grounding prompt used for single-step coordinate prediction
PROMPT_GROUNDING = (
    "Output only the coordinate of one point in your response. "
    "What element matches the following task: "
)


def build_messages(image_input: str, instruction_text: str, scene: str = "computer") -> List[Dict[str, Any]]:
    """Create a chat-style message payload for the image-text-to-text pipeline, using
    UI-TARS-style prompts and correct image/text ordering.

    image_input can be:
      - An HTTP(S) URL string
      - A local file path to an image

    scene:
      - "computer": computer-use agent prompt, text + image
      - "grounding": single-step grounding prompt, image + text
    """
    if image_input.lower().startswith(("http://", "https://")):
        image_content: Dict[str, Any] = {"type": "image", "url": image_input}
    else:
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image path does not exist: {image_input}")
        image = Image.open(image_input).convert("RGB")
        # Resize using UI-TARS smart_resize policy (divisible by 28, within pixel bounds)
        h_bar, w_bar = smart_resize(image.height, image.width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        if (image.width, image.height) != (w_bar, h_bar):
            image = image.resize((w_bar, h_bar), Image.Resampling.LANCZOS)
        image_content = {"type": "image", "image": image}

    scene_lower = (scene or "computer").lower()
    if scene_lower == "grounding":
        # Follow README_v1: image first, then text for grounding
        prompt_text = PROMPT_GROUNDING + instruction_text
        content = [
            image_content,
            {"type": "text", "text": prompt_text},
        ]
    else:
        # Default computer-use style: text first, then image
        prompt_text = PROMPT_COMPUTER + instruction_text
        content = [
            {"type": "text", "text": prompt_text},
            image_content,
        ]

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return int((number + factor - 1) // factor * factor)


def floor_by_factor(number: int, factor: int) -> int:
    return int(number // factor * factor)


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Mirror UI-TARS smart_resize: dimensions divisible by factor, pixels within bounds, aspect ratio preserved.
    Returns (h_bar, w_bar).
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        import math
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        import math
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)


def _load_processor_with_fallback(model_id: str):
    try:
        return AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
    except Exception as e:
        # Retry with explicit size override expected by Qwen2VLImageProcessor in newer Transformers
        try:
            return AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                size={"shortest_edge": 448, "longest_edge": 896},
                use_fast=True,
            )
        except Exception:
            raise e


def _run_inference_low_level(model_id: str, messages):
    processor = _load_processor_with_fallback(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)

    prompt_len = inputs["input_ids"].shape[-1]
    return processor.decode(outputs[0][prompt_len:]).strip()


def run_inference(model_id: str, image_input: str, instruction_text: str, max_new_tokens: int, scene: str = "computer") -> str:
    messages = build_messages(image_input, instruction_text, scene=scene)
    try:
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            trust_remote_code=True,
        )
        outputs = pipe(text=messages, max_new_tokens=max_new_tokens)

        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict):
                return (first.get("generated_text") or first.get("text") or str(first)).strip()
            return str(first).strip()
        return str(outputs).strip()
    except Exception:
        # Fallback to low-level processor+model path
        return _run_inference_low_level(model_id, messages)


# CLI interface for standalone testing
def main() -> None:
    """CLI interface for testing the model directly"""
    parser = argparse.ArgumentParser(description="UI-TARS-2B-SFT image+instruction → text")
    parser.add_argument("--image", required=True, help="Image URL or local image path")
    parser.add_argument("--instruction", required=True, help="Instruction or question for the model")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Model ID on Hugging Face Hub")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum new tokens to generate")
    args = parser.parse_args()

    result_text = run_inference(
        model_id=args.model_id,
        image_input=args.image,
        instruction_text=args.instruction,
        max_new_tokens=args.max_new_tokens,
    )
    print(result_text)


if __name__ == "__main__":
    main()


