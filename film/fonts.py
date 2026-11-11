"""Font lookup on the render machine (macOS keeps CJK fonts in downloadable assets)."""
import glob
from functools import lru_cache
from PIL import ImageFont

SYS = "/System/Library/Fonts/"
SUP = SYS + "Supplemental/"
# not installed until someone asks for them: Font Book, search, Download
ON_DEMAND = {"Kaiti.ttc": "Kaiti SC (楷体-简)", "Xingkai.ttc": "Xingkai SC (行楷-简)"}


@lru_cache(maxsize=None)
def path(name: str) -> str:
    for base in (SYS, SUP):
        hits = glob.glob(base + name)
        if hits:
            return hits[0]
    hits = glob.glob(f"/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/{name}")
    if hits:
        return hits[0]
    if name in ON_DEMAND:
        raise FileNotFoundError(f"{name}: open Font Book, search for {ON_DEMAND[name]} and click Download")
    raise FileNotFoundError(name)


@lru_cache(maxsize=None)
def font(name: str, size: int, index: int = 0):
    return ImageFont.truetype(path(name), size, index=index)
