import torch
import argparse
import numpy as np

from head_detector import HeadDetector
from PIL import Image


"""
python demo.py \
    --image_path="image.jpg" \
    --result_path="result.png"
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    detector = HeadDetector(device=device, model="vgg_heads_l")

    image = Image.open(args.image_path)
    image = image.convert("RGB")
    image = np.array(image)

    pncc, _ = detector(image)
    pncc = Image.fromarray(pncc)
    pncc.save(args.result_path)


if __name__ == "__main__":
    main()
