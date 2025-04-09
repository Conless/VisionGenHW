import json
import os
import time
import requests
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup
import tqdm

def get_pixiv_info(pid: int) -> None | Tuple[str, List[str]]:
    url = f"https://www.pixiv.net/artworks/{pid}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Fail to fetch: {response.status_code}")
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    meta_preload = soup.find("meta", {"id": "meta-preload-data"})
    if meta_preload is None:
        print("Fail to find meta-preload-data tag.")
        return None

    try:
        data = json.loads(meta_preload.get("content")) # type: ignore
    except json.JSONDecodeError as e:
        print("Fail to parse JSON:", e)
        return None

    illust_data = data.get("illust", {}).get(str(pid))
    if illust_data is None:
        print(f"Fail to find artwork with {pid = }")
        return None

    title = illust_data.get("illustTitle", "")
    tags_info = illust_data.get("tags", {}).get("tags", [])
    tags = [tag.get("tag", "") for tag in tags_info]

    return title, tags

def test_run():
    info = get_pixiv_info(pid = 128996811)
    assert info is not None
    title, tags = info
    print(f"{title = }, {tags = }")

def main():
    cache_result: Dict[int, None | Tuple[str, List[str]]] = {}

    def _parse(name: str):
        pid, _ = name.split("_")
        pid = int(pid)
        if pid not in cache_result:
            cache_result[pid] = get_pixiv_info(pid)
            time.sleep(0.1) # don't spam the server
        return cache_result[pid]

    final_result: List[Tuple[str, str, List[str]]] = []

    for name in tqdm.tqdm(os.listdir("data/pixiv/")):
        if not name.endswith((".jpg", ".png")):
            continue
        try:
            if result := _parse(name):
                final_result.append((name, *result))
        except Exception as e:
            print(f"Error parsing {name}: {e}")
            continue

    for name, title, tags in final_result:
        name = name.removesuffix(".png")
        name = name.removesuffix(".jpg")
        with open(f"data/pixiv/{name}.txt", "w") as f:
            f.write(f"{title}\n{tags}\n")

if __name__ == "__main__":
    main()
