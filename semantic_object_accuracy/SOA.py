import argparse
import os
import pickle as pkl
import warnings

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from tqdm import tqdm
from util import get_label, load_classes, load_file


setup_logger()
warnings.filterwarnings("ignore")

# Load Mask R-CNN object detector
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "weights/coco_mask_rcnn_detector.pkl"
predictor = DefaultPredictor(cfg)


def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images", dest="images", help="Image/Directory containing images to perform detection upon", type=str
    )
    parser.add_argument(
        "--detected_results",
        dest="detected_results",
        help="Image/Directory to store detections to",
        default="output",
        type=str,
    )
    parser.add_argument("--saved_file", dest="saved_file", help="File to store scores to", type=str)
    return parser.parse_args()


def run_mask_rcnn(args):
    images = args.images

    # check that the given folder contains exactly 80 folders
    _all_dirs = os.listdir(images)
    _num_folders = 0
    for _dir in _all_dirs:
        if os.path.isdir(os.path.join(images, _dir)):
            _num_folders += 1
    if _num_folders != 80:
        print("")
        print("****************************************************************************")
        print("\tWARNING")
        print("\tDid not find exactly 80 folders ({} folders found) in {}.".format(_num_folders, images))
        print(
            "\tFor the final calculation please make sure the folder {} contains one subfolder for each of the labels.".format(
                images
            )
        )
        print("\tCalculating scores on {}/80 labels now, but results will not be conclusive.".format(_num_folders))
        print("****************************************************************************")

    if not os.path.exists(args.detected_results):
        os.makedirs(args.detected_results)

    classes = load_classes("weights/coco.names")

    # go through all folders of generated images
    for dir in tqdm(os.listdir(images)):
        full_dir = os.path.join(images, dir)

        # check if detection was already run for this label
        if os.path.isfile(os.path.join(args.detected_results, "detected_{}.pkl".format(dir))):
            print("Detection already run for {}. Continuing with next label.".format(dir))
            continue

        image_list = os.listdir(full_dir)
        output_dict = {}

        # get MaskRCNN predictions for images in current folder
        for filename in image_list:
            img = cv2.imread(os.path.join(full_dir, filename))
            preds = predictor(img)
            # preds = non_max_suppression(preds, confidence, nms_thresh)
            class_ids = preds["instances"].pred_classes.cpu().numpy()
            boxes = preds["instances"].pred_boxes.tensor.cpu().numpy()

            img_preds_name = []
            img_preds_id = []
            img_bboxs = []

            if len(class_ids) > 0:
                for idx, cls_id in enumerate(class_ids):
                    pred_name = classes[cls_id]
                    box = boxes[idx]

                    img_preds_id.append(cls_id)
                    img_preds_name.append(pred_name)
                    img_bboxs.append(box)
                output_dict[filename.split("/")[-1]] = [img_preds_name, img_preds_id, img_bboxs]

        with open(os.path.join(args.detected_results, "detected_{}.pkl".format(dir)), "wb") as f:
            pkl.dump(output_dict, f)


def calc_recall(predicted_bbox, label):
    """Calculate how often a given object (label) was detected in the images"""
    correctly_recognized = 0
    num_images_total = len(predicted_bbox.keys())
    for key in predicted_bbox.keys():
        predictions = predicted_bbox[key]
        for recognized_label in predictions[1]:
            if recognized_label == label:
                correctly_recognized += 1
                break
    if num_images_total == 0:
        return 0, 0, 0
    accuracy = float(correctly_recognized) / num_images_total
    return accuracy, correctly_recognized, num_images_total


def calc_overall_class_average_accuracy(dict):
    """Calculate SOA-C"""
    accuracy = 0
    for label in dict.keys():
        accuracy += dict[label]["accuracy"]
    overall_accuracy = accuracy / len(dict.keys())
    return overall_accuracy


def calc_image_weighted_average_accuracy(dict):
    """Calculate SOA-I"""
    accuracy = 0
    total_images = 0
    for label in dict.keys():
        num_images = dict[label]["images_total"]
        accuracy += num_images * dict[label]["accuracy"]
        total_images += num_images
    overall_accuracy = accuracy / total_images
    return overall_accuracy


def calc_split_class_average_accuracy(dict):
    """Calculate SOA-C-Top/Bot-40"""
    num_img_list = []
    for label in dict.keys():
        num_img_list.append([label, dict[label]["images_total"]])
    num_img_list.sort(key=lambda x: x[1])
    sorted_label_list = [x[0] for x in num_img_list]

    bottom_40_accuracy = 0
    top_40_accuracy = 0
    for label in dict.keys():
        if sorted_label_list.index(label) < 40:
            bottom_40_accuracy += dict[label]["accuracy"]
        else:
            top_40_accuracy += dict[label]["accuracy"]
    bottom_40_accuracy /= 0.5 * len(dict.keys())
    top_40_accuracy /= 0.5 * len(dict.keys())

    return top_40_accuracy, bottom_40_accuracy


def calc_soa(args):
    """Calculate SOA scores"""
    results_dict = {}

    # find detection results
    mask_rcnn_detected_files = [
        os.path.join(args.detected_results, _file)
        for _file in os.listdir(args.detected_results)
        if _file.endswith(".pkl") and _file.startswith("detected_")
    ]

    # go through mask_rcnn detection and check how often it detected the desired object (based on the label)
    for mask_rcnn_file in mask_rcnn_detected_files:
        mask_rcnn = load_file(mask_rcnn_file)
        label = get_label(mask_rcnn_file)
        acc, correctly_recog, num_imgs_total = calc_recall(mask_rcnn, label)

        results_dict[label] = {}
        results_dict[label]["accuracy"] = acc
        results_dict[label]["images_recognized"] = correctly_recog
        results_dict[label]["images_total"] = num_imgs_total

    # calculate SOA-C and SOA-I
    print("")
    class_average_acc = calc_overall_class_average_accuracy(results_dict)
    print("Class average accuracy for all classes (SOA-C) is: {:6.4f}".format(class_average_acc))

    image_average_acc = calc_image_weighted_average_accuracy(results_dict)
    print("Image weighted average accuracy (SOA-I) is: {:6.4f}".format(image_average_acc))

    top_40_class_average_acc, bottom_40_class_average_acc = calc_split_class_average_accuracy(results_dict)
    print(
        "Top (SOA-C-Top40) and Bottom (SOA-C-Bot40) 40 class average accuracy is: {:6.4f} and {:6.4f}".format(
            top_40_class_average_acc, bottom_40_class_average_acc
        )
    )

    # store results
    with open(os.path.join(args.detected_results, "result_file.pkl"), "wb") as f:
        pkl.dump(results_dict, f)

    with open(os.path.join(args.saved_file), "w") as f:
        f.write("Class average accuracy for all classes (SOA-C) is: {:6.4f} \n".format(class_average_acc))
        f.write("Image weighted average accuracy (SOA-I) is: {:6.4f} \n".format(image_average_acc))
        f.write(
            "Top (SOA-C-Top40) and Bottom (SOA-C-Bot40) 40 class average accuracy is: {:6.4f} and {:6.4f}".format(
                top_40_class_average_acc, bottom_40_class_average_acc
            )
        )


if __name__ == "__main__":
    args = arg_parse()

    # use MaskRCNN on all images
    print("Using MaskRCNN Network on Generated Images...")
    run_mask_rcnn(args)

    # calculate score
    print("Calculating SOA Score...")
    calc_soa(args)
