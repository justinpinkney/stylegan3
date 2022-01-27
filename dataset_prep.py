from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
import sys
from joblib import Parallel, delayed

base_dir = Path(sys.argv[1])
new_dir = Path(sys.argv[2])
assert base_dir.is_dir(), f"Not a directory: {base_dir}"
assert not new_dir.is_dir(), f"Already exists: {new_dir}"

new_dir.mkdir()

def process_image(im_path, do_crop=False):
    im = Image.open(im_path).convert("RGB")
    # make a 256x256 version
    im_256 = im.resize((256, 256))
    stem = im_path.stem
    im_256.save(new_dir/f"{stem}_256.png")

    # and a random full size crop version
    if do_crop:
        loc = (random.randint(0,768), random.randint(0,768))
        box = [loc[0], loc[1], loc[0]+256, loc[1]+256]
        im_crop = im.crop(box)
        im_crop.save(new_dir/f"{stem}_crop.png")

im_paths = list(base_dir.glob("*.png"))
print(f"Found {len(im_paths)} files")


Parallel(n_jobs=32)(
    delayed(process_image)(im_path) for im_path in tqdm(im_paths)
    )
   
