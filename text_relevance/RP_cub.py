import argparse
import os
import pickle
import warnings

import numpy as np
import torch
import torchvision.transforms as transforms
from encoders import CNN_ENCODER, RNN_ENCODER
from nltk.tokenize import RegexpTokenizer
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Calculating R-precision")
    parser.add_argument("--image_dir", default="", type=str, help="Path to the folder containing generated images.")
    parser.add_argument("--rp_input_file", default="captions/CUB_RP_captions.pkl", type=str)
    parser.add_argument("--saved_file_path", default=None, type=str, help="Path to file saving result")
    parser.add_argument("--gpu_id", default="0", type=str)

    args = parser.parse_args()
    return args


args = parse_args()

# Load vocab data
with open("../text_to_images_models/data/birds/captions.pickle", "rb") as f:
    x = pickle.load(f, encoding="latin1")
    ixtoword, wordtoix = x[2], x[3]
    del x
    n_words = len(ixtoword)

# Load Text Encoder
text_encoder = RNN_ENCODER(n_words, nhidden=256)
state_dict = torch.load(
    "../text_to_images_models/DAMSMencoders/bird/text_encoder200.pth", map_location=lambda storage, loc: storage
)
text_encoder.load_state_dict(state_dict)
text_encoder.eval()
text_encoder.cuda()

# Load Image Encoder
image_encoder = CNN_ENCODER(256)
state_dict = torch.load(
    "../text_to_images_models/DAMSMencoders/bird/image_encoder200.pth", map_location=lambda storage, loc: storage
)
image_encoder.load_state_dict(state_dict)
image_encoder.eval()
image_encoder.cuda()


def tokennize_and_preprocess_captions(sents):
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

    captions = Variable(torch.from_numpy(captions), volatile=True)
    cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
    captions = captions.cuda()
    cap_lens = cap_lens.cuda()

    return captions, cap_lens


def r_precision_for_one_sample(gt_caption, mis_captions, image):

    gt_caption, gt_cap_len = tokennize_and_preprocess_captions([gt_caption])
    mis_captions, mis_cap_lens = tokennize_and_preprocess_captions(mis_captions)

    # --> Extract text embeddings #
    # GT caption
    hidden = text_encoder.init_hidden(1)
    _, gt_rnn_code = text_encoder(gt_caption, gt_cap_len, hidden)

    # False captions
    hidden = text_encoder.init_hidden(99)
    _, mis_rnn_code = text_encoder(mis_captions, mis_cap_lens, hidden)

    rnn_code = torch.cat((gt_rnn_code, mis_rnn_code), 0)

    # --> Extract image embeddings #
    img = image.unsqueeze(0)
    img = img.cuda()

    _, cnn_code = image_encoder(img)

    # cnn_code = 1 * nef
    # rnn_code = 100 * nef
    scores = torch.mm(cnn_code, rnn_code.transpose(0, 1))  # 1 * 100
    cnn_code_norm = torch.norm(cnn_code, 2, dim=1, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
    norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
    scores0 = scores / norm.clamp(min=1e-8)

    if torch.argmax(scores0) == 0:
        return 1

    return 0


img_transform = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

with open(args.rp_input_file, "rb") as f:
    RP_input = pickle.load(f)
RP_scores = []

count = 0
for test_item in tqdm(RP_input):
    caption = test_item["caption"]
    caption_id = test_item["caption_id"]
    mismatched_captions = test_item["mismatched_captions"]
    gen_image = Image.open(os.path.join(args.image_dir, f"{caption_id}.png")).convert("RGB")
    gen_image = img_transform(gen_image)
    rp = r_precision_for_one_sample(caption, mismatched_captions, gen_image)
    RP_scores.append(rp)

bins = np.zeros(10)
np.random.shuffle(RP_scores)
for i in range(10):
    bins[i] = np.average(RP_scores[i * 3000 : (i + 1) * 3000 - 1])
R_mean = np.average(bins)
R_std = np.std(bins)

# Save results to file
with open(f"{args.saved_file_path}", "w") as f:
    f.write("R mean:{:.6f} std:{:.6f}".format(R_mean, R_std))

print("R mean:{:.6f} std:{:.6f}".format(R_mean, R_std))
