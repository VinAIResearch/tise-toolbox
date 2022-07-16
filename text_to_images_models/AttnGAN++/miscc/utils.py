import argparse
import errno
import os
from copy import deepcopy

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
from miscc.config import cfg
from PIL import Image, ImageDraw, ImageFont


def str2bool(v):
    return v.lower() in ("true")


def get_parameters():

    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--dataset", type=str, default="birds", choices=["birds", "coco"])
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_models", type=int, default=0)
    parser.add_argument("--manualSeed", type=int, default=21)

    # Model Option
    parser.add_argument("--net_e", type=str, default="../DAMSMencoders/bird/text_encoder200.pth")
    parser.add_argument("--net_g", type=str, default="")

    # Smooth
    parser.add_argument("--smooth_gamma_1", type=float, default=4.0)
    parser.add_argument("--smooth_gamma_2", type=float, default=5.0)
    parser.add_argument("--smooth_gamma_3", type=float, default=10.0)
    parser.add_argument("--smooth_lambda", type=float, default=5.0)

    # Model hyper-parameters

    parser.add_argument("--gf_dim", type=int, default=64)
    parser.add_argument("--df_dim", type=int, default=32)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--condition_dim", type=int, default=100)
    parser.add_argument("--num_residual", type=int, default=2)
    parser.add_argument("--num_branch", type=int, default=3)
    parser.add_argument("--base_size", type=int, default=64)

    parser.add_argument("--rnn_type", type=str, default="LSTM")
    parser.add_argument("--text_emb_dim", type=int, default=256)
    parser.add_argument("--caps_per_img", type=int, default=10, help="CUB: 5, COCO: 10")
    parser.add_argument("--words_num", type=int, default=18)

    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--d_lr", type=float, default=2e-4)
    parser.add_argument("--g_lr", type=float, default=2e-4)
    parser.add_argument("--encoder_lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rnn_grad_clip", type=float, default=0.25)
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--eval", type=str2bool, default=False)

    # Path
    parser.add_argument("--checkpoint_path", type=str, default="/vinai/tandm3/text2image-train/checkpoints")
    parser.add_argument("--output_res_dir", type=str, default="/home/ubuntu/text2image-train/checkpoints")
    parser.add_argument("--samples_dir", type=str, default="../samples")
    parser.add_argument("--data_dir", type=str, default="../data/birds")
    parser.add_argument("--encoders_dir", type=str, default="../DAMSMencoders")

    # Step size
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=100)
    parser.add_argument("--snapshot_interval", type=int, default=5)

    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


# For visualization ################################################
COLOR_DIC = {
    0: [128, 64, 128],
    1: [244, 35, 232],
    2: [70, 70, 70],
    3: [102, 102, 156],
    4: [190, 153, 153],
    5: [153, 153, 153],
    6: [250, 170, 30],
    7: [220, 220, 0],
    8: [107, 142, 35],
    9: [152, 251, 152],
    10: [70, 130, 180],
    11: [220, 20, 60],
    12: [255, 0, 0],
    13: [0, 0, 142],
    14: [119, 11, 32],
    15: [0, 60, 100],
    16: [0, 80, 100],
    17: [0, 0, 230],
    18: [0, 0, 70],
    19: [0, 0, 0],
}
FONT_MAX = 50


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    fnt = ImageFont.truetype("./miscc/FreeMono.ttf", 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode("ascii", "ignore").decode("ascii")
            d.text(
                ((j + off1) * (vis_size + off2), i * FONT_MAX),
                "%d:%s" % (j, word[:6]),
                font=fnt,
                fill=(255, 255, 255, 255),
            )
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(
    real_imgs,
    captions,
    ixtoword,
    attn_maps,
    att_sze,
    lr_imgs=None,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    max_word_num=cfg.TEXT.WORDS_NUM,
):
    nvis = 8
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = np.ones([batch_size * FONT_MAX, (max_word_num + 2) * (vis_size + 2), 3], dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]

    real_imgs = nn.Upsample(size=(vis_size, vis_size), mode="bilinear")(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = nn.Upsample(size=(vis_size, vis_size), mode="bilinear")(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                one_map = skimage.transform.pyramid_expand(one_map, sigma=20, upscale=vis_size // att_sze)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255
                #
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = Image.new("RGBA", (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new("L", (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX : (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print("txt", txt.shape, "row", row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def build_super_images2(real_imgs, captions, cap_lens, ixtoword, attn_maps, att_sze, vis_size=256, topK=5):
    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    text_convas = np.ones([batch_size * FONT_MAX, max_word_num * (vis_size + 2), 3], dtype=np.uint8)

    real_imgs = nn.Upsample(size=(vis_size, vis_size), mode="bilinear")(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps)

    text_map, sentences = drawCaption(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2.0 / float(num_attn)
        #
        img = real_imgs[i]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []
        for j in range(num_attn):
            one_map = attn[j]
            mask0 = one_map > (2.0 * thresh)
            conf_score.append(np.sum(one_map * mask0))
            mask = one_map > thresh
            one_map = one_map * mask
            if (vis_size // att_sze) > 1:
                one_map = skimage.transform.pyramid_expand(one_map, sigma=20, upscale=vis_size // att_sze)
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map)
        sorted_indices = np.argsort(conf_score)[::-1]

        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255
            #
            PIL_im = Image.fromarray(np.uint8(img))
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged = Image.new("RGBA", (vis_size, vis_size), (0, 0, 0, 0))
            mask = Image.new("L", (vis_size, vis_size), (180))  # (210)
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3]

            row.append(np.concatenate([one_map, middle_pad], 1))
            #
            row_merge.append(np.concatenate([merged, middle_pad], 1))
            #
            txt = text_map[i * FONT_MAX : (i + 1) * FONT_MAX, j * (vis_size + 2) : (j + 1) * (vis_size + 2), :]
            row_txt.append(txt)
        # reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print("Warnings: txt", txt.shape, "row", row.shape, "row_merge_new", row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


####################################################################
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.orthogonal(m.weight.data, 1.0)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#     elif classname.find('Linear') != -1:
#         nn.init.orthogonal(m.weight.data, 1.0)
#         if m.bias is not None:
#             m.bias.data.fill_(0.0)


def weights_init(m):
    # orthogonal_
    # xavier_uniform_
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # print(m.state_dict().keys())
        if list(m.state_dict().keys())[0] == "weight":
            nn.init.orthogonal_(m.weight.data, 1.0)
        elif list(m.state_dict().keys())[3] == "weight_bar":
            nn.init.orthogonal_(m.weight_bar.data, 1.0)
        # nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
