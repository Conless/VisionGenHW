from diffusers import FluxPipeline # type: ignore
from typing import List
import torch

@torch.inference_mode()
def merge_and_save(
    lora_paths: List[str],
    output_paths: List[str],
    base_model: str = "black-forest-labs/FLUX.1-dev",
    device: str = "cuda"
) -> None:
    assert len(lora_paths) == len(output_paths)
    pipe = FluxPipeline.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).to(device)
    for lora_path, output_path in zip(lora_paths, output_paths):
        pipe.load_lora_weights(lora_path, adapter_name="flux-lora")
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()
        pipe.save_pretrained(output_path)

if __name__ == "main":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python merge_lora.py <lora_path> <output_path>")
        sys.exit(1)
    lora_path = sys.argv[2]
    output_path = sys.argv[3]
    merge_and_save([lora_path], [output_path])
