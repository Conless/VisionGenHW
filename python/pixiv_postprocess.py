import os
from typing import Dict, List, Set, Tuple
import tqdm

def read_tag_remap() -> Dict[str, str]:
    with open("remap.py", "r") as f:
        return eval(f.read())

def main():
    visited_tags: Set[str] = set()
    def _parse(title: str) -> Tuple[str, str, str, List[str]]:
        with open(f"data/pixiv/{title}", "r") as f:
            data = f.readlines()
        title = data[0]
        tags = [tag.strip() for tag in eval(data[1])]
        desc = " ".join(data[2:]).strip()
        visited_tags.update(tags)
        return name, title, desc, tags

    file_data: List[Tuple[str, str, str, List[str]]] = []
    for name in tqdm.tqdm(os.listdir("data/pixiv/")):
        if not name.endswith(("txt")):
            continue
        try:
            file_data.append(_parse(name))
        except Exception as e:
            print(f"Error parsing {name}: {e}")
            continue

    remap = read_tag_remap()
    if set(remap.keys()) != visited_tags:
        print("Mismatch between visited tags and remap keys.")
        print("Please complete these tags in remap.py:")
        print(f"{visited_tags - remap.keys()}")
        print("Please remove these tags from remap.py:")
        print(f"{remap.keys() - visited_tags}")
        return

    def final_tags(tags: List[str]) -> str:
        tmp = set([remap[tag] for tag in tags if tag in remap])
        tmp -= {""}
        return "[" + ",".join(tmp) + "]"

    for name, title, desc, tags in tqdm.tqdm(file_data):
        desc = desc.replace("\n", " ")
        title = title.replace("\n", " ")
        name = name.replace(".txt", ".tmp")
        tags = final_tags(tags)
        content = f"[Artwork] {title}. [Tag] {tags} [Detail]: {desc}."
        with open(f"data/pixiv/{name}", "w") as f:
            f.write(content)

if __name__ == "__main__":
    main()
