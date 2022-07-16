import argparse
import math
import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nest import modules
from PIL import Image
from sklearn.metrics import mean_squared_error
from torchvision import transforms
from tqdm import tqdm


warnings.filterwarnings("ignore")

# COCO classes
nms = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Calculating Counting metric")
    parser.add_argument("--image_dir", default="", type=str)
    parser.add_argument("--ct_input_file", default="captions/CA_input_captions.pkl", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--result_file", default="", type=str)
    args = parser.parse_args()
    return args


args = parse_args()

gpu_id = args.gpu_id
torch.cuda.set_device(gpu_id)

# Image preprocessing
image_size = 448
transformer = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# Counting models
backbone = modules.fc_resnet50(channels=240, pretrained=False)

model = modules.peak_response_mapping(
    backbone, enable_peak_stimulation=True, peak_stimulation="addedmodule5", sub_pixel_locating_factor=1
)

# Loaded pre-trained weights
model = nn.DataParallel(model)
state = torch.load("weights/coco14.pt")
model.load_state_dict(state["model"])
model = model.module.cuda()
model = model.eval()

# Load counting data
print("Load data from: ", args.ct_input_file)
with open(args.ct_input_file, "rb") as f:
    counting_data = pickle.load(f)


# Get predicted counting from an image
def predict(img_path):
    raw_img = Image.open(img_path).convert("RGB")
    input_var = transformer(raw_img).unsqueeze(0).cuda().requires_grad_()
    confidence, density_map, _ = model(input_var, 1)
    count_den = F.adaptive_avg_pool2d(density_map, 1).squeeze(2).squeeze(2).detach().cpu().numpy()[0]
    density_map = density_map.squeeze().detach().cpu().numpy()
    confidence = confidence.cpu().detach().numpy()
    confidence[confidence < 0] = 0
    confidence = confidence[0]
    confidence[confidence > 0] = 1
    count = np.round(confidence * count_den)
    res = {}
    for index in range(len(nms)):
        if count[index]:
            res[nms[index]] = count[index]
    return res


# Compute CA metric
rmse_images = []
img_dir = args.image_dir
for item in tqdm(counting_data):
    image_path = os.path.join(img_dir, str(item["caption_id"]) + ".png")
    pred = predict(image_path)
    gt = item["counting_info"]
    gt_img = []
    pred_img = []
    for key in gt:
        if key in pred:
            gt_img.append(gt[key])
            pred_img.append(float(pred[key]))
        else:
            gt_img.append(gt[key])
            pred_img.append(0.0)
    rmse_img = math.sqrt(mean_squared_error(gt_img, pred_img))
    rmse_images.append(rmse_img)
avg_rmse_images = np.mean(rmse_images)

# Save result to file
with open(args.result_file, "w") as f:
    f.write(f"CA = {avg_rmse_images}")

# Log on terminal
print(f"CA = {avg_rmse_images}")
