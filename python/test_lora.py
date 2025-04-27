from diffusers import FluxPipeline # type: ignore
from typing import List
import torch

@torch.inference_mode()
def merge_and_save(
    lora_paths: List[str],
    output_paths: List[str],
    prompts: List[str],
    base_model: str = "black-forest-labs/FLUX.1-dev",
    device: str = "cuda"
) -> None:
    assert len(lora_paths) == len(output_paths) == len(prompts)
    pipe = FluxPipeline.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).to(device)
    for lora_path, output_path, prompt in zip(lora_paths, output_paths, prompts):
        print(f"Merging LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name="flux-lora")
        # then generate a image
        result = pipe(
            prompt,
            num_inference_steps=100,
        ).images[0]
        result.save(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python test_lora.py <lora_path> <output_path> <prompt>")
        sys.exit(1)
    lora_path = sys.argv[1]
    output_path = sys.argv[2]
    prompt = sys.argv[3]
    merge_and_save([lora_path], [output_path], [prompt])
