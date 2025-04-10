import csv
import os
import torch
from typing import List, Tuple
import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration # type: ignore
from PIL import Image

MODEL_NAME="fancyfeast/llama-joycaption-alpha-two-hf-llava"
PROMPT = "Write a brief caption, give keywords for this image in a formal tone."
processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="bfloat16", device_map=0)
llava_model.eval()
# Build the conversation
convo = [
    {
        "role": "system",
        "content": "You are a helpful image captioner.",
    },
    {
        "role": "user",
        "content": PROMPT,
    },
]

# Format the conversation
# WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
# but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
# if not careful, which can make the model perform poorly.
CONVI_STRING = processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
assert isinstance(CONVI_STRING, str)

@torch.inference_mode()
def process_batch(batch_size: int, files: List[str]) -> List[Tuple[str, str]]:
    # split the files into batches
    batches: List[List[str]] = []
    for i in range(0, len(files), batch_size):
        batches.append(files[i:min(i + batch_size, len(files))])

    # process each batch
    results: List[Tuple[str, str]] = []
    for i in tqdm.tqdm(range(len(batches))):
        batch = batches[i]
        images = []
        for file in batch:
            image = Image.open(path + file)
            images.append(image)
        # process the images
        inputs = processor(text=len(images)*[CONVI_STRING], images=images, return_tensors="pt").to('cuda')
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        # Generate the captions
        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=0.01,
            top_k=None,
            top_p=0.9,
        )
        assert isinstance(generate_ids, torch.Tensor)
        # Remove the input_ids from the generated ids
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        # Decode the caption
        captions = processor.batch_decode(generate_ids, skip_special_tokens=True)
        assert len(captions) == len(batch)
        for file, caption in zip(batch, captions):
            results.append((file, caption))

    return results

@torch.inference_mode()
def process(batch_size: int = 1, path = "pixiv/") -> List[Tuple[str, str]]:
    assert path.endswith("/")
    required_files: List[str] = []
    for file in tqdm.tqdm(os.listdir(path)):
        if not file.endswith((".jpg", ".png")):
            continue
        required_files.append(file)
    print(f"Found {len(required_files)} files in {path}.")
    return process_batch(batch_size, required_files)


def write_csv(info: List[Tuple[str, str]], path = "pixiv/", exist_ok=False) -> None:
    assert path.endswith("/")
    csv_file = f"{path}metadata.csv"
    if os.path.exists(csv_file) and not exist_ok:
        raise FileExistsError(f"{csv_file} already exists. Set exist_ok=True to overwrite it.")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "caption"])
        for name, caption in info:
            writer.writerow([name, caption])

if __name__ == "__main__":
    BATCH_SIZE = 16
    path = "dataset/"
    # time based random seed
    torch.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    if not os.path.exists(f"{path}metadata.csv"):
        result = process(batch_size=BATCH_SIZE, path=path)
        write_csv(result, path=path, exist_ok=False)
    else:
        # first read csv, get the file names
        result: List[Tuple[str, str]] = []
        with open(f"{path}metadata.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                result.append((row[0], row[1]))
        # make a map
        result_map = {k: v for k, v in result}
        recompute = []
        for k, v in result_map.items():
            if not v or str(v).strip() == "":
                recompute.append(k)
        if not recompute:
            print("No empty captions found.")
        else:
            print(f"{len(recompute)} captions to be recomputed.")
            new_result = process_batch(batch_size=BATCH_SIZE, files=recompute)
            # update the result_map with the new captions
            for name, caption in new_result:
                assert name in result_map
                result_map[name] = caption
            # write the new result_map to the csv
            write_csv(list(result_map.items()), path=path, exist_ok=True)
