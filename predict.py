import argparse
import json
import os
import glob

import numpy as np
import torch
from PIL import Image

import detectron2
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from mask2former import add_maskformer2_config

# Setup detectron2 logger
setup_logger()


def find_all_images_in_dir(images_dir: str):
    image_files = []
    for extension in ["jpg", "png"]:
        image_files += glob.glob(f"{images_dir}/*.{extension}")
    return image_files


def setup_cfg(config_file: str, checkpoint_file: str,):
    # Create base config
    cfg = get_cfg()
    # Add deeplab and mask2former configs
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    # Add model specific configs
    cfg.merge_from_file(config_file)
    # Configure weights
    cfg.MODEL.WEIGHTS = checkpoint_file
    return cfg


def predict(
    config_file: str,
    checkpoint_file: str,
    images_dir: str,
    output_dir: str,
):
    assert os.path.isfile(config_file)
    assert os.path.isfile(checkpoint_file)
    assert os.path.isdir(images_dir)
    os.makedirs(output_dir, exist_ok=True)

    cpu_device = torch.device("cpu")

    # Create Detectron2 config
    cfg = setup_cfg(config_file, checkpoint_file)

    # Create default predictor
    predictor = DefaultPredictor(cfg)

    # Collect all the images in the target dir
    images_files = find_all_images_in_dir(images_dir)

    # Run inference over every image
    for image_file in images_files:
        image = read_image(image_file, format="BGR")
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        # Save the segmentation as png
        panoptic_seg, segments_info = predictor(image)["panoptic_seg"]
        panoptic_seg_file = os.path.join(
            output_dir,
            image_id + "_segmentation.png",
        )
        panoptic_seg_image = panoptic_seg.to(cpu_device).numpy()
        Image.fromarray(panoptic_seg_image).save(panoptic_seg_file)
        # Save the segments info as json
        segments_info_file = os.path.join(
            output_dir,
            image_id + "_segments_info.json",
        )

        with open(segments_info_file, 'w') as f:
            json.dump(segments_info, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with given models on the selected images."
    )

    parser.add_argument(
        "--config_file",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--checkpoint_file",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict(
        config_file=args.config_file,
        checkpoint_file=args.checkpoint_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
    )
