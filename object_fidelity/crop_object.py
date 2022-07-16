import argparse
import os
import warnings

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from PIL import Image
from tqdm import tqdm


setup_logger()
warnings.filterwarnings("ignore")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "./weights/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)


def crop_object(predictor, src_dir, dest_dir):
    filenames = os.listdir(src_dir)
    count = 0
    thing_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    for filename in tqdm(filenames):
        img = cv2.imread(os.path.join(src_dir, filename))
        outputs = predictor(img)
        class_ids = outputs["instances"].pred_classes.cpu().numpy()
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        for idx, cls_id in enumerate(class_ids):
            obj_img = img.crop(boxes[idx])
            width, height = obj_img.size
            class_name = thing_classes[cls_id]

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            obj_img.save(os.path.join(dest_dir, filename.split(".")[0] + "_{}_{}.png".format(class_name, count)))
            count += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_image_dir", default="", type=str)
    parser.add_argument("--saved_cropped_object_dir", default="", type=str)
    args = parser.parse_args()
    return args


args = parse_args()

# Crop objects
crop_object(predictor, args.source_image_dir, args.saved_cropped_object_dir)
