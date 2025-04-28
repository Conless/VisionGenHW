from diffusers import FluxPipeline # type: ignore
from typing import List
import torch

@torch.inference_mode()
def merge_and_save(
    output_paths: List[str],
    prompts: List[str],
    base_model: str = "black-forest-labs/FLUX.1-dev",
    device: str = "cuda"
) -> None:
    assert len(output_paths) == len(prompts)
    pipe = FluxPipeline.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).to(device)
    for output_path, prompt in zip(output_paths, prompts):
        # then generate a image
        result = pipe(
            prompt,
            num_inference_steps=50,
        ).images[0] # type: ignore
        result.save(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python test_model.py <model_path> <output_path> <prompt>")
        sys.exit(1)
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    prompt = sys.argv[3]
    merge_and_save([output_path], [prompt], base_model=model_path)
