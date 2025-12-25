import os
import os.path
import sys
import time
sys.path.append('../../manipulator_grasp')

import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
import threading
import spatialmath as sm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from manipulator_grasp.env.dynamic_y import Yenv
from manipulator_grasp.env.dynamic_yz import YZEnv
from manipulator_grasp.env.rotate_env import RotateEnv

from PIL import Image

def save_rgb(img: np.ndarray, out_path: str) -> None:
    """Save RGB numpy image to out_path (png). Handles uint8 or float images."""
    rgb = img
    if rgb is None:
        raise RuntimeError("imgs['img'] is None")
    # drop alpha if exists
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    # convert float->uint8 if needed
    if rgb.dtype != np.uint8:
        mx = float(np.max(rgb))
        if mx <= 1.5:  # likely 0~1
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:          # already 0~255-ish
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(rgb).save(out_path)

def main() -> None:
    env = YZEnv()
    env.reset()
    try:
        # save one RGB snapshot to output/
        out_dir = os.path.join(ROOT_DIR, "output")
        os.makedirs(out_dir, exist_ok=True)
        imgs = env.render()
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"rgb_{ts}.png")
        save_rgb(imgs["img"], out_path)
        print("[showenv] saved:", out_path)
        env.hold()
    finally:
        env.close()

if __name__ == '__main__':
    main()
