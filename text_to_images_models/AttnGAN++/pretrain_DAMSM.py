import argparse
import logging
import os
import pprint
import random
import sys
import time

# For ignore user warnings
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from datasets import TextDataset, prepare_data
from encoders import CNN_ENCODER, RNN_ENCODER
from miscc.config import cfg, cfg_from_file
from miscc.losses import sent_loss, words_loss
from miscc.utils import build_super_images, mkdir_p
from PIL import Image
from torch.autograd import Variable


warnings.filterwarnings("ignore", category=UserWarning)

dir_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "./."))
sys.path.append(dir_path)

UPDATE_INTERVAL = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DAMSM network")
    parser.add_argument("--cfg", dest="cfg_file", help="optional config file", default="cfg/DAMSM/bird.yml", type=str)
    parser.add_argument("--manualSeed", type=int, help="manual seed")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--gpu", dest="gpu_id", type=int, default=0)
    parser.add_argument("--data_dir", dest="data_dir", type=str, default="")
    parser.add_argument("--model_path", dest="model_path", type=str, default="")
    parser.add_argument("--checkpoint_path", dest="checkpoint_path", type=str, default="")
    parser.add_argument("--pretrained_model", type=int, default=0)
    args = parser.parse_args()
    return args


def log_func(logger, log_info):
    logger.info(log_info)  # Log to file
    print(log_info)  # Print to console


def train(dataloader, cnn_model, rnn_model, batch_size, labels, optimizer, epoch, ixtoword, image_dir, logger):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    competition_loss_total = 0

    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader, 0):
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs[-1])
        # --> batch_size x nef x 17*17
        _, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn_maps, competition_loss = words_loss(
            words_features, words_emb, labels, cap_lens, class_ids, batch_size
        )
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        competition_loss_total += competition_loss
        loss = w_loss0 + w_loss1 + cfg.TRAIN.SMOOTH.ALPHA_1 * competition_loss

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        #
        loss.backward()

        # Gradient clipping for prevent exploding gradient in RNNs/LSTM
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if (step + 1) % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step + 1

            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

            competition_cur_loss = competition_loss_total.item() / UPDATE_INTERVAL

            cur_damsm_loss = s_cur_loss0 + s_cur_loss1 + w_cur_loss0 + w_cur_loss1
            cur_acm_loss = cur_damsm_loss + cfg.TRAIN.SMOOTH.ALPHA_1 * competition_cur_loss

            elapsed = time.time() - start_time

            # Log info
            log_info = "| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | s_loss {:7.4f} | w_loss {:7.4f} | damsm_loss {:7.4f} | comp_loss {:7.4f} | acm_loss {:7.4f} | ".format(
                epoch,
                step,
                len(dataloader),
                elapsed * 1000.0 / UPDATE_INTERVAL,
                s_cur_loss0 + s_cur_loss1,
                w_cur_loss0 + w_cur_loss1,
                cur_damsm_loss,
                competition_cur_loss,
                cur_acm_loss,
            )

            log_func(logger, log_info)

            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            competition_loss_total = 0

            start_time = time.time()

            # attention Maps
            img_set, _ = build_super_images(imgs[-1].cpu(), captions, ixtoword, attn_maps, att_sze)

            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = "%s/attention_maps%d.png" % (image_dir, epoch)
                im.save(fullpath)

    return count


def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    comp_total_loss = 0

    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

        words_features, sent_code = cnn_model(real_imgs[-1])
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn, comp_loss = words_loss(
            words_features, words_emb, labels, cap_lens, class_ids, batch_size
        )
        w_total_loss += (w_loss0 + w_loss1).data
        comp_total_loss += comp_loss.data

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / (step + 1)
    w_cur_loss = w_total_loss.item() / (step + 1)
    comp_cur_loss = comp_total_loss.item() / (step + 1)

    return s_cur_loss, w_cur_loss, comp_cur_loss


def build_models(model_dir, pretrained_model, init_lr):
    # ==> Build model
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    # ==> Parameters
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    optimizer = optim.Adam(para, lr=init_lr, betas=(0.5, 0.999))
    labels = Variable(torch.LongTensor(range(batch_size)))

    if pretrained_model > 0:
        # ===> Resume training

        start_epoch = pretrained_model + 1
        # Load checkpoint for training
        checkpoint_path = "%s/checkpoints_epoch_%d.pth" % (model_dir, pretrained_model)
        checkpoint = torch.load(checkpoint_path)

        # Load text_encoder
        text_encoder.load_state_dict(checkpoint["text_encoder"])
        print("Load pretrained text-encoder at: ", checkpoint_path)

        # Load image_encoder
        image_encoder.load_state_dict(checkpoint["image_encoder"])
        print("Load pretrained image-encoder at ", checkpoint_path)

        # Load optimizer
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Load optimizer at ", checkpoint_path)

        # Load other info for resume training
        training_info = {
            "cur_best_acm_loss": checkpoint["cur_best_acm_loss"],
            "cur_best_damsm_loss": checkpoint["cur_best_damsm_loss"],
            "cur_best_epoch": checkpoint["cur_best_epoch"],
            "cur_best_sent_loss": checkpoint["cur_best_sent_loss"],
            "cur_best_word_loss": checkpoint["cur_best_word_loss"],
            "lr": checkpoint["lr"],
        }
    else:
        # ==> Training from scratch
        start_epoch = 0
        training_info = {
            "cur_best_acm_loss": float("inf"),
            "cur_best_damsm_loss": float("inf"),
            "cur_best_epoch": 0,
            "cur_best_sent_loss": float("inf"),
            "cur_best_word_loss": float("inf"),
            "lr": lr,
        }

    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return optimizer, text_encoder, image_encoder, labels, start_epoch, training_info


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.VERSION = args.version
    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != "":
        cfg.DATA_DIR = args.data_dir
    print("Using config:")
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

    ##########################################################################
    model_path = "%s/%s_%s" % (args.model_path, cfg.DATASET_NAME, cfg.VERSION)
    checkpoint_path = "%s/%s_%s" % (args.checkpoint_path, cfg.DATASET_NAME, cfg.VERSION)
    if not os.path.exists(os.path.join(checkpoint_path, "DAMSM_train_history.log")):
        open(os.path.join(checkpoint_path, "DAMSM_train_history.log"), "a").close()

    #############################################################################
    # Setting for logger
    logger = logging.getLogger("damsm")
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler(os.path.join(checkpoint_path, "DAMSM_train_history.log"))
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    #############################################################################

    model_dir = os.path.join(model_path, "Model_DAMSM")
    image_dir = os.path.join(checkpoint_path, "Image_DAMSM")
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose(
        [transforms.Scale(int(imsize * 76 / 64)), transforms.RandomCrop(imsize), transforms.RandomHorizontalFlip()]
    )
    dataset = TextDataset(cfg.DATA_DIR, "train", base_size=cfg.TREE.BASE_SIZE, transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS)
    )

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, "test", base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS)
    )

    # Train ##############################################################
    lr = cfg.TRAIN.ENCODER_LR
    optimizer, text_encoder, image_encoder, labels, start_epoch, training_info = build_models(
        model_dir, args.pretrained_model, lr
    )

    try:
        cur_best_acm_loss = training_info["cur_best_acm_loss"]
        cur_best_damsm_loss = training_info["cur_best_damsm_loss"]
        cur_best_epoch = training_info["cur_best_epoch"]
        cur_best_word_loss = training_info["cur_best_word_loss"]
        cur_best_sent_loss = training_info["cur_best_sent_loss"]
        lr = training_info["lr"]
        print(training_info)

        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            epoch_start_time = time.time()

            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            print(optimizer)

            count = train(
                dataloader,
                image_encoder,
                text_encoder,
                batch_size,
                labels,
                optimizer,
                epoch,
                dataset.ixtoword,
                image_dir,
                logger,
            )

            log_func(logger, "-" * 170)
            if len(dataloader_val) > 0:
                s_loss, w_loss, comp_loss = evaluate(dataloader_val, image_encoder, text_encoder, batch_size)
                damsm_loss = s_loss + w_loss
                acm_loss = damsm_loss + cfg.TRAIN.SMOOTH.ALPHA_1 * comp_loss

                log_info = "| end epoch {:3d} | valid_s_loss {:7.4f} | valid_w_loss {:7.4f} | DAMSM_loss  {:7.4f} | ACM_loss  {:7.4f} | lr {:.5f}|".format(
                    epoch + 1, s_loss, w_loss, damsm_loss, acm_loss, lr
                )
                log_func(logger, log_info)
            log_func(logger, "-" * 170)

            # Learning rate decay
            if lr > cfg.TRAIN.ENCODER_LR / 10.0:
                lr *= 0.98

            # Save best model
            if acm_loss < cur_best_acm_loss:
                cur_best_acm_loss = acm_loss
                cur_best_damsm_loss = damsm_loss
                cur_best_word_loss = w_loss
                cur_best_sent_loss = s_loss

                # Save model
                torch.save(image_encoder.state_dict(), "%s/best_image_encoder.pth" % (model_dir))
                torch.save(text_encoder.state_dict(), "%s/best_text_encoder.pth" % (model_dir))

                # Log info
                log_func(logger, "*" * 170)
                log_func(logger, "Save best image encoder and text encoder!")
                log_info = "| cur_best_acm_loss {:7.4f} | cur_best_damsm {:7.4f} | cur_word_loss {:7.4f} | cur_sent_loss {:7.4f} | cur_best_epoch {:3d}".format(
                    cur_best_acm_loss, cur_best_damsm_loss, cur_best_word_loss, cur_best_sent_loss, cur_best_epoch
                )
                log_func(logger, log_info)
                log_func(logger, "*" * 170)

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                training_info = {
                    "cur_best_acm_loss": cur_best_acm_loss,
                    "cur_best_damsm_loss": cur_best_damsm_loss,
                    "cur_best_epoch": cur_best_epoch,
                    "cur_best_sent_loss": cur_best_sent_loss,
                    "cur_best_word_loss": cur_best_word_loss,
                    "lr": lr,
                }
                checkpoint_to_save = {
                    "image_encoder": image_encoder.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    **training_info,
                }

                # Save checkpoints
                torch.save(checkpoint_to_save, "%s/checkpoints_epoch_%d.pth" % (model_dir, epoch))
                print("Save checkpoints at epoch {}!".format(epoch))

                # Remove previous checkpoint model for save memory
                previous_epoch = epoch - cfg.TRAIN.SNAPSHOT_INTERVAL
                pre_checkpoint_path = "%s/checkpoints_epoch_%d.pth" % (model_dir, previous_epoch)
                if os.path.exists(pre_checkpoint_path):
                    os.remove(pre_checkpoint_path)
                    log_info = "Removed previous checkpoint at {}".format(pre_checkpoint_path)
                    log_func(logger, log_info)

            if epoch == cfg.TRAIN.MAX_EPOCH:
                log_info = "| best_acm {:7.4f} | best_damsm {:7.4f} | word_loss {:7.4f} | sent_loss {:7.4f} | best_epoch {:3d}".format(
                    cur_best_acm_loss, cur_best_damsm_loss, cur_best_word_loss, cur_best_sent_loss, cur_best_epoch
                )
                log_func(logger, log_info)

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
