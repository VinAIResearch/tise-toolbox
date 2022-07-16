from __future__ import print_function

import logging
import os
import pprint
import random
import sys

# For ignore warnings
import warnings

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import TextDataset
from miscc.config import cfg, update_config
from miscc.utils import get_parameters, mkdir_p
from nltk.tokenize import RegexpTokenizer
from trainer import Trainer


warnings.filterwarnings("ignore", category=UserWarning)

dir_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "./."))
sys.path.append(dir_path)


def gen_example(wordtoix, algo):
    """generate images from example sentences"""
    filepath = "%s/example_filenames.txt" % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().split("\n")
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = "%s/%s.txt" % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print("Load from:", name)
                sentences = f.read().split("\n")
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r"\w+")
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print("sent", sent)
                        continue

                    print(sent)
                    rev = []
                    for t in tokens:
                        t = t.encode("ascii", "ignore").decode("ascii")
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
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
            key = name[(name.rfind("/") + 1) :]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


# Entry point
if __name__ == "__main__":
    # Configurations
    args = get_parameters()
    update_config(args)

    if cfg.CUDA:
        torch.backends.cudnn.benchmark = True

    print("Using configuration:")
    pprint.pprint(cfg)

    # Fix random seed
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.set_device(args.gpu_id)
        torch.cuda.manual_seed_all(args.manualSeed)

    checkpoint_model_dir = "%s/%s_%s" % (args.checkpoint_path, cfg.DATASET_NAME, cfg.VERSION)
    output_res_dir = "%s/%s_%s" % (args.output_res_dir, cfg.DATASET_NAME, cfg.VERSION)

    # Create checkpoint dir
    mkdir_p(checkpoint_model_dir)
    mkdir_p(output_res_dir)

    open(os.path.join(output_res_dir, "train_history.log"), "a").close()
    r_precision_path = os.path.join(output_res_dir, "R_precision.txt")

    split_dir, shuffle = "train", True

    #############################################################################
    # Setting for logger
    logger = logging.getLogger("dm_gan")
    logger.setLevel(logging.INFO)

    f_handler = logging.FileHandler(os.path.join(output_res_dir, "train_history.log"))
    f_handler.setLevel(logging.INFO)

    logger.addHandler(f_handler)
    #############################################################################

    if not cfg.TRAIN.FLAG:
        split_dir = "test"
        shuffle = False

    # Dataset and Dataloader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose(
        [transforms.Scale(int(imsize * 76 / 64)), transforms.RandomCrop(imsize), transforms.RandomHorizontalFlip()]
    )

    dataset = TextDataset(cfg.DATA_DIR, split_dir, base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=shuffle, num_workers=int(cfg.WORKERS)
    )

    algo = Trainer(
        checkpoint_model_dir,
        output_res_dir,
        dataloader,
        dataset.n_words,
        dataset.ixtoword,
        args.pretrained_models,
        logger,
        dataset,
        args.version,
    )

    if cfg.TRAIN.FLAG:
        # Training
        algo.train()
    else:
        # Eval
        if cfg.EVAL:
            algo.sampling(split_dir, cfg.TRAIN.NET_G, cfg.PATH.SAMPLES, r_precision_path)
        else:
            # Generating samples from customize caption.
            gen_example(dataset.wordtoix, algo)
