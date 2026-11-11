"""Font lookup on the render machine (macOS keeps CJK fonts in downloadable assets)."""
import glob
from functools import lru_cache
from PIL import ImageFont

SYS = "/System/Library/Fonts/"
SUP = SYS + "Supplemental/"


@lru_cache(maxsize=None)
def path(name: str) -> str:
    for base in (SYS, SUP):
        hits = glob.glob(base + name)
        if hits:
            return hits[0]
    hits = glob.glob(f"/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/{name}")
    if hits:
        return hits[0]
    raise FileNotFoundError(name)


@lru_cache(maxsize=None)
def font(name: str, size: int, index: int = 0):
    return ImageFont.truetype(path(name), size, index=index)
