import argparse
import os
import pickle
import warnings

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="", type=str, help="Path to the folder containing generated images.")
    parser.add_argument("--pa_input_file", default="captions/PA_input_captions.pkl", type=str)
    parser.add_argument("--saved_file_path", default=None, type=str, help="Path to file saving result")
    parser.add_argument("--gpu_id", default="0", type=str)

    args = parser.parse_args()
    return args


args = parse_args()

device = f"cuda:{args.gpu_id}"
model, preprocess = clip.load("ViT-B/32", device=device)


def test(gt_caption, mis_caption, img_path):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text = clip.tokenize([gt_caption] + [mis_caption]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        if probs[0] > 0.6:
            return 1.0
    return 0.0


with open(args.pa_input_file, "rb") as f:
    data = pickle.load(f)

phrase_res = dict().fromkeys(data.keys())
for p in phrase_res:
    phrase_res[p] = {"success": 0.0, "total": 0.0, "score": 0.0}

i = 0
for phrase in data:
    for item in tqdm(data[phrase]):
        phrase_res[phrase]["success"] += test(
            item["caption"],
            item["false_caption"],
            os.path.join(args.image_dir, phrase, str(item["caption_id"]) + ".png"),
        )
        phrase_res[phrase]["total"] += 1
        phrase_res[phrase]["score"] = phrase_res[phrase]["success"] / phrase_res[phrase]["total"]
        i += 1
    print(phrase, phrase_res[phrase])


PA = np.mean([phrase_res[i]["score"] for i in phrase_res])

# Save result to file
with open(args.saved_file_path, "w") as f:
    f.write(f"PA = {PA}")

# Log on terminal
print(f"PA = {PA}")
