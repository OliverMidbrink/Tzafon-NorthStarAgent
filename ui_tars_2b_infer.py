import argparse
import os
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForVision2Seq


DEFAULT_MODEL_ID = "ByteDance-Seed/UI-TARS-2B-SFT"


def build_messages(image_input: str, instruction_text: str) -> List[Dict[str, Any]]:
    """Create a chat-style message payload for the image-text-to-text pipeline.

    image_input can be:
      - An HTTP(S) URL string
      - A local file path to an image
    """
    if image_input.lower().startswith(("http://", "https://")):
        image_content: Dict[str, Any] = {"type": "image", "url": image_input}
    else:
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image path does not exist: {image_input}")
        image = Image.open(image_input).convert("RGB")
        image_content = {"type": "image", "image": image}

    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": instruction_text},
            ],
        }
    ]


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


def run_inference(model_id: str, image_input: str, instruction_text: str, max_new_tokens: int) -> str:
    messages = build_messages(image_input, instruction_text)
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


def main() -> None:
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


