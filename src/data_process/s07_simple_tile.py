import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage
import skimage.io
from tqdm import tqdm


"""
https://www.kaggle.com/iafoss/panda-16x128x128-tiles
https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/155424
"""


def get_tiles(img, tile_size, n_tiles, mask, mode=0):
    t_sz = tile_size
    h, w, c = img.shape
    pad_h = (t_sz - h % t_sz) % t_sz + ((t_sz * mode) // 2)
    pad_w = (t_sz - w % t_sz) % t_sz + ((t_sz * mode) // 2)

    img2 = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )
    img3 = img2.reshape(img2.shape[0] // t_sz, t_sz, img2.shape[1] // t_sz, t_sz, 3)
    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, t_sz, t_sz, 3)

    mask3 = None
    if mask is not None:
        mask2 = np.pad(
            mask,
            [
                [pad_h // 2, pad_h - pad_h // 2],
                [pad_w // 2, pad_w - pad_w // 2],
                [0, 0],
            ],
            constant_values=0,
        )
        mask3 = mask2.reshape(
            mask2.shape[0] // t_sz, t_sz, mask2.shape[1] // t_sz, t_sz, 3
        )
        mask3 = mask3.transpose(0, 2, 1, 3, 4).reshape(-1, t_sz, t_sz, 3)

    n_tiles_with_info = (
        img3.reshape(img3.shape[0], -1).sum(1) < t_sz ** 2 * 3 * 255
    ).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(
            img3,
            [[0, n_tiles - len(img3)], [0, 0], [0, 0], [0, 0]],
            constant_values=255,
        )
        if mask is not None:
            mask3 = np.pad(
                mask3,
                [[0, n_tiles - len(mask3)], [0, 0], [0, 0], [0, 0]],
                constant_values=0,
            )
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    if mask is not None:
        mask3 = mask3[idxs]
    return img3, mask3, n_tiles_with_info >= n_tiles


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--root-path", type=str, default="../input/")
    arg("--tile-num", type=int, default=64)
    arg("--tile-size", type=int, default=192)
    arg("--res-level", type=int, default=1, help="0:High, 1:Middle, 2:Low")
    arg("--resize", type=int, default=None)
    arg("--mode", type=int, default=0)
    return parser.parse_args()


def main():

    args = make_parse()
    n_tiles = args.tile_num
    tile_size = args.tile_size
    res_level = args.res_level
    resize_size = args.resize
    mode = args.mode

    root = Path(args.root_path)
    img_dir = root / "train_images"
    mask_dir = root / "train_label_masks"
    train = pd.read_csv(root / "train.csv")

    out_dir = (
        root / f"numtile-{n_tiles}-tilesize-{tile_size}-res-{res_level}-mode-{mode}"
    )
    out_dir.mkdir(exist_ok=True)
    train_out_dir = out_dir / "train"
    mask_out_dir = out_dir / "masks"
    train_out_dir.mkdir(exist_ok=True)
    mask_out_dir.mkdir(exist_ok=True)
    x_tot, x2_tot = [], []

    for img_id in tqdm(train.image_id):
        img_path = str(img_dir / (img_id + ".png"))
        mask_path = mask_dir / (img_id + "_mask.png")

        # image saved in BGR, cv2 imread get image in RGB
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path))  # mask saved in RGB
        tiles, masks, _ = get_tiles(
            img=image, tile_size=tile_size, n_tiles=n_tiles, mask=mask, mode=mode
        )

        if resize_size is not None:
            tiles = [cv2.resize(t, (resize_size, resize_size)) for t in tiles]
            masks = [cv2.resize(m, (resize_size, resize_size)) for m in masks]

        for idx, img in enumerate(tiles):
            # RGB
            x_tot.append((img / 255.0).reshape(-1, 3).mean(0))
            x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))

            # if read with PIL RGB turns into BGR
            # We get CRC error when unzip if not cv2.imencode
            # img = cv2.imencode(".png", image)[1]
            # img_out.writestr(f"{img_id}_{idx}.png", img)
            cv2.imwrite(f"{str(train_out_dir)}/{img_id}_{idx}.png", img)

            # mask[:, :, 0] has value in {0, 1, 2, 3, 4, 5}, other mask is 0 only
            # if mask is not None:
            #     mask = masks[idx]
            #     cv2.imwrite(f"{str(mask_out_dir)}/{img_id}_{idx}.png", mask[:, :, 0])
            #     mask = cv2.imencode(".png", mask[:, :, 0])[1]
            #     mask_out.writestr(f"{img_id}_{idx}.png", mask)

    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    print("mean:", img_avr, ", std:", np.sqrt(img_std))


if __name__ == "__main__":
    main()
