""" Some parts of code are borrowed from: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py"""
import argparse
import os

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from scipy.stats import entropy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import models, transforms
from tqdm import tqdm


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    num_classes = 80
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.AuxLogits.fc = nn.Linear(768, num_classes)
    inception_model.fc = nn.Linear(2048, num_classes)
    inception_model.load_state_dict(torch.load("weights/inceptionv3_fine_to_with_80_coco_classes.pth"))
    inception_model.type(dtype)
    inception_model.eval()

    up = nn.Upsample(size=(299, 299), mode="bilinear").type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        temperature = 2.1737587451934814
        x = x / temperature
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, num_classes))

    for i, batch in enumerate(tqdm(dataloader), 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, imgspath):
        self.imgspath = imgspath
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.namelist = os.listdir(self.imgspath)

    def __getitem__(self, index):
        imgname = self.namelist[index]
        img_path = os.path.join(self.imgspath, imgname)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.namelist)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="", type=str)
    parser.add_argument("--saved_file", default="", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    args = parser.parse_args()
    return args


args = parse_args()

print("Load images from: ", args.image_dir)
imgs = IgnoreLabelDataset(args.image_dir)

print("Calculating Inception Score...")
torch.cuda.set_device(args.gpu_id)
IS_mean, IS_std = inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=10)

# Save result to file
with open(args.saved_file, "w") as f:
    f.write(f"O-IS: {IS_mean} +-  {IS_std}")

print(f"O-IS: {IS_mean} +- {IS_std}")
