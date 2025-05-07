import argparse
from dataclasses import dataclass
from diffusers import FluxPipeline # type: ignore
import os
import torch

@dataclass
class RuntimeConfig:
    steps: int
    height: int
    width: int

@torch.inference_mode()
def evaluate(pipe: FluxPipeline, dir: str, file: str, config: RuntimeConfig) -> None:
    with open(file, "r") as f:
        prompts = f.readlines()
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()
        if not prompt:
            continue
        result = pipe(
            prompt,
            num_inference_steps=config.steps,
            height=config.height,
            width=config.width,
        )
        image = result.images[0] # type: ignore
        image.save(os.path.join(dir, f"{i}.png"))

@lambda f: f() if __name__ == "__main__" else None
@torch.inference_mode()
def _():
    parser = argparse.ArgumentParser(description="Evaluate a batch of images.")
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to the directory containing the texts to evaluate.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Path to the LoRA weights to use for evaluation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model to use for evaluation.",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to the LoRA weights to use for evaluation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps to use.",
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=512,
        help="Height of the generated images.",
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=512,
        help="Width of the generated images.",
    )
    parser.add_argument(
        "--enable-input", "--stdin",
        action="store_true",
        help="Use stdin to get the input.",
    )

    args = parser.parse_args()
    data: str           = args.data
    model: str          = args.model
    lora: str | None    = args.lora
    output: str         = args.output_dir
    enable_input: bool  = args.enable_input

    args.__dict__.pop("data", None)
    args.__dict__.pop("output_dir", None)
    args.__dict__.pop("lora", None)
    args.__dict__.pop("model", None)
    args.__dict__.pop("enable_input", None)
    config: RuntimeConfig = RuntimeConfig(**args.__dict__)

    if os.path.exists(output):
        print(f"Output directory {output} already exists. Please remove it before running the script.")
        return

    pipe = FluxPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    if lora is not None:
        pipe.load_lora_weights(lora)

    if enable_input: # input from stdin
        os.makedirs(output)
        print("Please enter the prompts (one per line). Press Ctrl+D to finish.")
        i = 0
        while True:
            try:
                prompt = input("Prompt: ")
                if not prompt:
                    break
                if prompt.startswith("--config "):
                    param = prompt.split(" ")
                    attr = param[1]
                    value = param[2]
                    if hasattr(config, attr):
                        old_value = getattr(config, attr)
                        print(f"Changing {attr} from {old_value} to {value}")
                        setattr(config, attr, eval(value))
                    else:
                        print(f"Invalid config parameter: {attr}")
                    continue
                result = pipe(
                    prompt,
                    num_inference_steps=config.steps,
                    height=config.height,
                    width=config.width,
                )
                image = result.images[0] # type: ignore
                file = os.path.join(output, f"{i}.png")
                print(f"Saving image to {file}")
                image.save(file)
                i += 1
            except EOFError:
                break
        return

    for x, _, files in os.walk(data):
        cache_dir = os.path.join(output, os.path.basename(x))
        os.makedirs(cache_dir)
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(x, file)
                final_dir = os.path.join(cache_dir, file[:-4])
                print(f"Evaluating {file_path} with model {model} and lora {lora}")
                os.makedirs(final_dir)
                evaluate(pipe, final_dir, file_path, config)
