import os
import torch
from typing import List
import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor # type: ignore
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", use_fast=True)

def get_input(abs_path: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file:///{abs_path}",
                },
                {"type": "text", "text": "Briefly describe this image in English."},
            ],
        }
    ]
    # Preparation for inference
    text: str = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return process_vision_info(messages), text # type: ignore

def process_batch(vision_info, texts: List[str]):
    image_inputs, video_inputs = vision_info
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.01)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

@torch.inference_mode()
def process(batch_size: int = 1):
    pwd = os.path.dirname(os.path.abspath(__file__))
    batch = []
    def run_batch():
        if len(batch) == 0:
            return []
        image_inputs = []
        video_inputs = None
        for b in batch:
            image, _ = b[0]
            image_inputs += image
        text_inputs = [b[1] for b in batch]
        result = process_batch((image_inputs, video_inputs), text_inputs)
        batch.clear()
        assert len(result) == len(text_inputs)
        ret = []
        for t, b in zip(result, batch):
            ret.append((b[2], t))
        return ret

    final_results = []

    for file in tqdm.tqdm(os.listdir("data")):
        if not file.endswith((".jpg", ".png")):
            continue

        abs_path = abs_file = str(os.path.join(pwd, "data", file))
        abs_path = abs_path.replace(".jpg", "")
        abs_path = abs_path.replace(".png", "")

        vision_info, text = get_input(abs_file)
        batch.append((vision_info, text, abs_path))
        if len(batch) == batch_size:
            final_results += run_batch()
    return final_results + run_batch()
