"""Make a fully disjoint, reproducible real-photograph transfer benchmark.

Photographs are loaded from the scikit-image sample gallery at test time and
ARE NOT committed to this repository. Training and test image identities differ.
Requires: pip install scikit-image pillow numpy
"""
from pathlib import Path
import tempfile
import sys
import numpy as np
from PIL import Image
from skimage import data

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "modelimage-v3-photos")
TRAIN = ["astronaut", "camera", "coins", "brick", "grass", "moon", "immunohistochemistry", "retina"]
TEST = ["coffee", "chelsea", "rocket", "hubble_deep_field"]

def image(name):
    a = np.asarray(getattr(data, name)())
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    if a.ndim == 2:
        a = np.repeat(a[:,:,None], 3, axis=2)
    if a.shape[2] == 4:
        a = a[:,:,:3]
    return Image.fromarray(a.astype(np.uint8), "RGB")

def slices(img, split, seed, count):
    width, height = img.size
    rng = np.random.default_rng(seed)
    side = min(width, height)
    for j in range(count):
        fraction = float(rng.uniform(.42,.82))
        edge = max(96,int(side*fraction))
        x = int(rng.integers(0,max(1,width-edge+1)))
        y = int(rng.integers(0,max(1,height-edge+1)))
        crop=img.crop((x,y,x+edge,y+edge)).resize((96,96),Image.Resampling.BICUBIC)
        if j%3==2: crop=crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        yield crop

def run():
    for split, names, count in [("train",TRAIN,12),("test",TEST,3)]:
        dest=OUT/split
        dest.mkdir(parents=True,exist_ok=True)
        for i,name in enumerate(names):
            img=image(name)
            for j,part in enumerate(slices(img,split,20261022+i*107+(0 if split=="train" else 10000),count)):
                part.save(dest/f"{name}_{j:02d}.ppm")
    print(f"Real photos: {len(TRAIN)*12} training crops from {len(TRAIN)} photos; "
          f"{len(TEST)*3} test crops from {len(TEST)} DIFFERENT photos.")
    print(f"Saved to {OUT}")

if __name__=="__main__":
    run()
