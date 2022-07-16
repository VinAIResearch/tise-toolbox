import argparse
import os
import pickle
import random
import warnings

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculating R-precision")
    parser.add_argument("--image_dir", default="", type=str, help="Path to the folder containing generated images.")
    parser.add_argument("--rp_input_file", default="captions/COCO_RP_captions.pkl", type=str)
    parser.add_argument("--saved_file_path", default=None, type=str, help="Path to file saving result")
    parser.add_argument("--gpu_id", default="0", type=str)

    args = parser.parse_args()
    return args


args = parse_args()

device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open(args.rp_input_file, "rb") as f:
    RP_input = pickle.load(f)

image_dir = args.image_dir

# Divide all captions to 10 bins.
# Then, we run RP calculation process for each bin to get RP value for each bin.
# Finally, we compute mean and std of all RP values of bins to get final RP score.
num_captions = len(RP_input)
caption_item_ids = list(range(num_captions))
random.shuffle(caption_item_ids)
num_bins = 10
samples_per_bin = int(len(caption_item_ids) / num_bins)
bins = []
for i in range(num_bins):
    if i == (num_bins - 1) and num_captions % num_bins != 0:
        b = caption_item_ids[i * samples_per_bin :]
    else:
        b = caption_item_ids[i * samples_per_bin : (i + 1) * samples_per_bin]
    bins.append(b)

# Calculate RP value for each bin
RP_scores = []
for bin_idx, b in tqdm(enumerate(bins)):
    count = 0
    success_count = 0

    for item_idx in tqdm(b):
        count += 1
        # Image
        image_path = os.path.join(image_dir, str(RP_input[item_idx]["caption_id"]) + ".png")
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        # Text
        mis_captions = RP_input[item_idx]["mismatched_captions"]
        true_caption = RP_input[item_idx]["caption"]
        captions = [true_caption] + mis_captions
        text = clip.tokenize(captions).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            if np.argmax(probs) == 0:
                success_count += 1

    RP_bin_score = success_count * 1.0 / len(b)
    RP_scores.append(RP_bin_score)
    print(f"Bin: {bin_idx}, RP: {RP_bin_score}")

# Get final RP values
RP_mean = np.mean(RP_scores)
RP_std = np.std(RP_scores)
print(f"R-precision: {RP_mean} +- {RP_std}")

# Log results to file if you need
if args.saved_file_path is not None:
    with open(args.saved_file_path, "w") as f:
        f.write(f"R-precision: {RP_mean} +- {RP_std}")
