import argparse
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import TextDataset
from easydict import EasyDict as edict
from encoders import RNN_ENCODER
from generators import G_NET
from miscc.config import cfg, cfg_from_file
from nltk.tokenize import RegexpTokenizer
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning)


class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model, device_ids=None):
        super(_CustomDataParallel, self).__init__(model, device_ids=device_ids)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# Configurations
args = {}
args["cfg_file"] = "./cfg/eval_bird.yml"
args["gpu_id"] = 0
args["manualSeed"] = 100
args = edict(args)

if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

# Fix random seed
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if cfg.CUDA:
    torch.cuda.set_device(args.gpu_id)
    torch.cuda.manual_seed_all(args.manualSeed)

# Dataset
imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
image_transform = transforms.Compose(
    [transforms.Scale(int(imsize * 76 / 64)), transforms.RandomCrop(imsize), transforms.RandomHorizontalFlip()]
)

split_dir = "test"
dataset = TextDataset(cfg.DATA_DIR, split_dir, base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

# Load Text Encoder
text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)

# Load Pretrained Generator
netG = G_NET()
if cfg.TRAIN.NET_G.split("/")[-1].split("_")[0] == "coco":
    netG = _CustomDataParallel(netG, device_ids=[args.gpu_id])
state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
netG.load_state_dict(state_dict)

# For using GPU
if cfg.CUDA:
    text_encoder = text_encoder.cuda()
    netG.cuda()

# Set eval mode for inference
text_encoder.eval()
netG.eval()

wordtoix = dataset.wordtoix
ixtoword = dataset.ixtoword


# Generate img from batch captions
def generate_img_from_caption(sents):
    captions = []
    cap_lens = []
    for sent in sents:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print("sent: ", sent)
            continue
        rev = []
        for t in tokens:
            t = t.encode("ascii", "ignore").decode("ascii")
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))

    # sort batch caption
    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype="int64")
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    captions = cap_array

    batch_size = captions.shape[0]
    nz = cfg.GAN.Z_DIM
    captions = Variable(torch.from_numpy(captions), volatile=True)
    cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)

    if cfg.CUDA:
        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
        noise = noise.cuda()

    # --> Extract text embeddings
    hidden = text_encoder.init_hidden(batch_size)
    # words_embs: batch_size x nef x seq_len
    # sent_emb: batch_size x nef
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    mask = captions == 0

    noise.data.normal_(0, 1)
    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)

    generated_imgs = [None] * batch_size

    for j in range(batch_size):
        # For fake images
        ims = []
        for i1 in range(len(fake_imgs)):
            im = fake_imgs[i1][j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            ims.append(im)
        generated_imgs[sorted_indices[j]] = ims

    return generated_imgs


# ---------------------------------------------------------------------------------#


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_input_file", default="", type=str)
    parser.add_argument("--saved_dir", default="", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    args = parser.parse_args()
    return args


args = parse_args()

if not os.path.exists(args.saved_dir):
    os.makedirs(args.saved_dir)

with open(args.caption_input_file, "rb") as f:
    caption_data = pickle.load(f)

captions = list(zip([item["caption_id"] for item in caption_data], [item["caption"] for item in caption_data]))
num_captions = len(captions)
print("Number of captions: ", num_captions)

# Batch data
if num_captions % args.batch_size != 0:
    num_batches = int((num_captions / args.batch_size + 1))
else:
    num_batches = int(num_captions / args.batch_size)

caption_batches = []
for i in range(num_batches):
    if i != num_batches - 1:
        cur_batch = captions[i * args.batch_size : (i + 1) * args.batch_size]
    else:
        cur_batch = captions[i * args.batch_size :]
    caption_batches.append(cur_batch)

for cap_batch in tqdm(caption_batches):
    cap_ids = [c[0] for c in cap_batch]
    caps = [c[1] for c in cap_batch]
    images = generate_img_from_caption(caps)
    for cap_id, img in zip(cap_ids, images):
        img[-1].save(os.path.join(args.saved_dir, f"{cap_id}.png"))
