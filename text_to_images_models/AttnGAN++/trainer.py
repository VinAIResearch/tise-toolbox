from __future__ import print_function

import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import prepare_data
from discriminators import D_NET64, D_NET128, D_NET256
from encoders import CNN_ENCODER, RNN_ENCODER
from generators import G_NET
from miscc.config import cfg
from miscc.losses import KL_loss, discriminator_loss, generator_loss, words_loss
from miscc.utils import build_super_images, copy_G_params, count_parameters, load_params, mkdir_p, weights_init
from PIL import Image
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model, device_ids=None):
        super(_CustomDataParallel, self).__init__(model, device_ids=device_ids)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Trainer(object):
    def __init__(
        self, model_dir, output_res_dir, data_loader, n_words, ixtoword, pretrained_models, logger, dataset, version
    ):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(model_dir, "Model")
            self.image_dir = os.path.join(output_res_dir, "Image")
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        self.pretrained_models = pretrained_models
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.logger = logger
        self.dataset = dataset
        self.tensorboard_writer = SummaryWriter(f"../checkpoints/tfboard_logs/{version}")

    def build_models(self):
        # ============================ ENCODER ====================================#
        if cfg.TRAIN.NET_E == "":
            print("Error: no pretrained text-image encoders")
            return
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace("text_encoder", "image_encoder")
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print("Load Image-Encoder from:", img_encoder_path)
        image_encoder.eval()

        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)

        for p in text_encoder.parameters():
            p.requires_grad = False
        print("Load Text-Encoder from:", cfg.TRAIN.NET_E)
        text_encoder.eval()

        # ====================================================================#
        # --------------------- Generator and Discriminators------------------#
        # ====================================================================#
        print("==============================================================")
        print("------------------ Building models ---------------------------")
        print("==============================================================")
        # =====================================================================#
        netsD = []
        netG = G_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            netsD.append(D_NET64())
        if cfg.TREE.BRANCH_NUM > 1:
            netsD.append(D_NET128())
        if cfg.TREE.BRANCH_NUM > 2:
            netsD.append(D_NET256())
        # ======================================================================#
        # Initialize weights
        netG.apply(weights_init)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)

        print("Total parameters:")
        print("Generator: {}".format(count_parameters(netG)))
        D_total_params = 0
        for i in range(len(netsD)):
            num_params = count_parameters(netsD[i])
            D_total_params += num_params
            print("  * D_{}: {}".format(i, num_params))
        print("Discriminators: {}".format(D_total_params))

        start_epoch = 1
        # ======================================================================#
        # Load pretrained model
        if self.pretrained_models > 0:
            netG, netsD = self.load_pretrained_models(netG, netsD)
            start_epoch = self.pretrained_models + 1
        # ======================================================================#

        # For using GPUs
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()

        # Data Parallel for using multiple GPUs
        device_ids = [0, 1, 2]
        image_encoder = _CustomDataParallel(image_encoder, device_ids=device_ids)
        netG = _CustomDataParallel(netG, device_ids=device_ids)
        for i in range(len(netsD)):
            netsD[i] = _CustomDataParallel(netsD[i], device_ids=device_ids)

        return [text_encoder, image_encoder, netG, netsD, start_epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)

        # Optimizers for network D
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))

            # Load optimizer checkpoint
            opt_path = os.path.join(self.model_dir, "optim_D{}_epoch_{}.pth".format(i, self.pretrained_models))
            if self.pretrained_models > 0 and os.path.exists(opt_path):
                opt.load_state_dict(torch.load(opt_path, map_location=lambda storage, loc: storage))
                print("Load optimizer of net_D{} successfully! epoch ({}) ".format(i, self.pretrained_models))
            optimizersD.append(opt)

        # Optimizers for network G
        optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))

        # Load optimizer checkpoint
        opt_path = os.path.join(self.model_dir, "optim_G_epoch_{}.pth".format(self.pretrained_models))
        if self.pretrained_models > 0 and os.path.exists(opt_path):
            optimizerG.load_state_dict(torch.load(opt_path, map_location=lambda storage, loc: storage))
            print("Load optimizer of net_G successfully! epoch ({})".format(self.pretrained_models))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = torch.FloatTensor(batch_size).fill_(1)
        fake_labels = torch.FloatTensor(batch_size).fill_(0)
        match_labels = torch.LongTensor(range(batch_size))

        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_optim(self, optDs, optG, epoch):
        for i in range(len(optDs)):
            torch.save(optDs[i].state_dict(), "{}/optim_D{}_epoch_{}.pth".format(self.model_dir, i, epoch))
        torch.save(optG.state_dict(), "{}/optim_G_epoch_{}.pth".format(self.model_dir, epoch))
        print("Save optim checkpoints at {}".format(self.model_dir))

        # Remove previous checkpoint optim for save mem
        checkpoint_epoch = epoch - self.snapshot_interval
        for i in range(len(optDs)):
            opt_path = "{}/optim_D{}_epoch_{}.pth".format(self.model_dir, i, checkpoint_epoch)
            if os.path.exists(opt_path):
                os.remove(opt_path)
                print("Remove D{} optim checkpoints at {}".format(i, opt_path))

        opt_path = "{}/optim_G_epoch_{}.pth".format(self.model_dir, checkpoint_epoch)
        if os.path.exists(opt_path):
            os.remove(opt_path)
            print("Remove G optim checkpoints at {}".format(opt_path))

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(), "{}/netG_epoch_{}.pth".format(self.model_dir, epoch))
        load_params(netG, backup_para)
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(), "{}/netD{}_epoch{}.pth".format(self.model_dir, i, epoch))
        print("Save checkpoint model at {}".format(self.model_dir))

        # Remove previous model
        for i in range(len(netsD)):
            netD_path = "{}/netD{}_epoch{}.pth".format(self.model_dir, i, epoch - self.snapshot_interval)
            if os.path.exists(netD_path):
                os.remove(netD_path)
            print("Remove netD{} model at {}".format(i, self.model_dir))

    def load_pretrained_models(self, netG, netDs):
        netG.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, "netG_epoch_{}.pth".format(self.pretrained_models)),
                map_location=lambda storage, loc: storage,
            )
        )
        for i in range(len(netDs)):
            netDs[i].load_state_dict(
                torch.load(
                    os.path.join(self.model_dir, "netD{}_epoch{}.pth".format(i, self.pretrained_models)),
                    map_location=lambda storage, loc: storage,
                )
            )
        print("Loaded pretrained models (epoch: {}) successfully!".format(self.pretrained_models))
        return netG, netDs

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(
        self,
        netG,
        noise,
        sent_emb,
        words_embs,
        mask,
        image_encoder,
        captions,
        cap_lens,
        gen_iterations,
        name="current",
    ):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = build_super_images(img, captions, self.ixtoword, attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = "%s/G_%s_%d_%d.png" % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(
            region_features.detach(), words_embs.detach(), None, cap_lens, None, self.batch_size
        )
        img_set, _ = build_super_images(fake_imgs[i].detach().cpu(), captions, self.ixtoword, att_maps, att_sze)

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = "%s/D_%s_%d.png" % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        # Building models
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM

        # Prepare noise
        noise = torch.FloatTensor(batch_size, nz)
        fixed_noise = torch.FloatTensor(batch_size, nz).normal_(0, 1)

        # For using GPUs
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        # Prepair labels
        real_labels, fake_labels, match_labels = self.prepare_labels()

        print("Starting training ...")
        gen_iterations = 0
        for epoch in range(start_epoch, self.max_epoch + 1):
            start_t = time.time()
            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = captions == 0
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask, cap_lens)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = []
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(
                        netsD[i],
                        imgs[i],
                        fake_imgs[i],
                        sent_emb,
                        real_labels,
                        fake_labels,
                        self.tensorboard_writer,
                        gen_iterations,
                        i,
                    )
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs.append(errD.data)

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # Compute total loss for training G
                step += 1
                gen_iterations += 1

                # Do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = generator_loss(
                    netsD,
                    image_encoder,
                    fake_imgs,
                    real_labels,
                    words_embs,
                    sent_emb,
                    match_labels,
                    cap_lens,
                    class_ids,
                    self.tensorboard_writer,
                    gen_iterations,
                )
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs.append(kl_loss.data)
                # Backward and update parameters
                errG_total.backward()
                optimizerG.step()

                self.tensorboard_writer.add_scalars(
                    "Generator_Loss/Total_Loss",
                    {"kl_loss": kl_loss.data, "total_loss": errG_total.data},
                    gen_iterations,
                )

                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(
                        "D0_loss: {:.4f}, D1_loss: {:.4f}, D2_loss: {:.4f}, G0_loss: {:.4f}, G1_loss: {:.4f}, G2_loss: {:.4f}, word_loss: {:.4f}, sent_loss: {:.4f}, kl_loss: {:.4f}".format(
                            D_logs[0],
                            D_logs[1],
                            D_logs[2],
                            G_logs[0],
                            G_logs[1],
                            G_logs[2],
                            G_logs[3],
                            G_logs[4],
                            G_logs[5],
                        )
                    )

                    # Log to file
                    self.logger.info(
                        "D0_loss: {:.4f}, D1_loss: {:.4f}, D2_loss: {:.4f}, G0_loss: {:.4f}, G1_loss: {:.4f}, G2_loss: {:.4f}, word_loss: {:.4f}, sent_loss: {:.4f}, kl_loss: {:.4f}".format(
                            D_logs[0],
                            D_logs[1],
                            D_logs[2],
                            G_logs[0],
                            G_logs[1],
                            G_logs[2],
                            G_logs[3],
                            G_logs[4],
                            G_logs[5],
                        )
                    )

                # Save images
                if gen_iterations % 10000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    load_params(netG, backup_para)

            elapsed = time.time() - start_t
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("========================================================================")
            print(
                "elapsed [{}], [{}/{}], Ds_loss: {:.4f}, Gs_loss: {:.4f}, D0_loss: {:.4f}, D1_loss: {:.4f}, D2_loss: {:.4f}, G0_loss: {:.4f}, G1_loss: {:.4f}, G2_loss: {:.4f}, word_loss: {:.4f}, sent_loss: {:.4f}, kl_loss: {:.4f}".format(
                    elapsed,
                    epoch,
                    self.max_epoch,
                    errD_total.data,
                    errG_total.data,
                    D_logs[0],
                    D_logs[1],
                    D_logs[2],
                    G_logs[0],
                    G_logs[1],
                    G_logs[2],
                    G_logs[3],
                    G_logs[4],
                    G_logs[5],
                )
            )
            print("========================================================================")

            # Log to file
            self.logger.info("========================================================================")
            self.logger.info(
                "elapsed [{}], [{}/{}], Ds_loss: {:.4f}, Gs_loss: {:.4f}, D0_loss: {:.4f}, D1_loss: {:.4f}, D2_loss: {:.4f}, G0_loss: {:.4f}, G1_loss: {:.4f}, G2_loss: {:.4f}, word_loss: {:.4f}, sent_loss: {:.4f}, kl_loss: {:.4f}".format(
                    elapsed,
                    epoch,
                    self.max_epoch,
                    errD_total.data,
                    errG_total.data,
                    D_logs[0],
                    D_logs[1],
                    D_logs[2],
                    G_logs[0],
                    G_logs[1],
                    G_logs[2],
                    G_logs[3],
                    G_logs[4],
                    G_logs[5],
                )
            )
            self.logger.info("========================================================================")

            # Checkpoint model
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:

                # Sample some image
                self.save_img_results(
                    netG,
                    fixed_noise,
                    sent_emb,
                    words_embs,
                    mask,
                    image_encoder,
                    captions,
                    cap_lens,
                    epoch,
                    name="average",
                )

                # Save model
                self.save_model(netG, avg_param_G, netsD, epoch)

                # Save optim
                self.save_optim(optimizersD, optimizerG, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)
        self.save_optim(optimizersD, optimizerG, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir, split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = "%s/single_samples/%s/%s" % (save_dir, split_dir, filenames[i])
            folder = s_tmp[: s_tmp.rfind("/")]
            if not os.path.isdir(folder):
                print("Make a new folder: ", folder)
                mkdir_p(folder)

            fullpath = "%s_%d.jpg" % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir, model_path, sample_path, r_precision_path=None):
        if model_path == "":
            print("Error: the path for morels is not found!")
        else:
            if split_dir == "test":
                split_dir = "valid"

            device_ids = [int(cfg.GPU_ID)]
            # Build and load the generator
            netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG = _CustomDataParallel(netG, device_ids=device_ids)
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print("Load G from: ", model_path)
            netG.eval()

            # load text encoder
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print("Load text encoder from:", cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # load image encoder
            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace("text_encoder", "image_encoder")
            state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print("Load image encoder from:", img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()
            image_encoder = _CustomDataParallel(image_encoder, device_ids=device_ids)

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            # the path to save generated images
            # Path to save generated images
            sample_path = "{}/{}".format(sample_path, split_dir)
            mkdir_p(sample_path)

            cnt = 0
            R_count = 0
            R = np.zeros(30000)
            cont = True
            for ii in range(11):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                if cont is False:
                    break
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if cont is False:
                        break
                    if step % 100 == 0:
                        print("cnt: ", cnt)

                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = captions == 0
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
                    for j in range(batch_size):
                        s_tmp = "%s/single/%s" % (sample_path, keys[j])
                        folder = s_tmp[: s_tmp.rfind("/")]
                        if not os.path.isdir(folder):
                            # print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = "%s_s%d_%d.png" % (s_tmp, k, ii)
                        im.save(fullpath)

                    _, cnn_code = image_encoder(fake_imgs[-1])

                    for i in range(batch_size):
                        mis_captions, mis_captions_len = self.dataset.get_mis_caption(class_ids[i])
                        hidden = text_encoder.init_hidden(99)
                        _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
                        rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                        # cnn_code = 1 * nef
                        # rnn_code = 100 * nef
                        scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                        cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                        rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                        norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                        scores0 = scores / norm.clamp(min=1e-8)
                        if torch.argmax(scores0) == 0:
                            R[R_count] = 1
                        R_count += 1

                    if R_count >= 30000:
                        sum = np.zeros(10)
                        np.random.shuffle(R)
                        for i in range(10):
                            sum[i] = np.average(R[i * 3000 : (i + 1) * 3000 - 1])
                        R_mean = np.average(sum)
                        R_std = np.std(sum)
                        print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))

                        if r_precision_path is not None:
                            if not os.path.exists(r_precision_path):
                                open(r_precision_path, "a").close()
                            with open(r_precision_path, "a") as f:
                                f.write("{} | R mean:{:.4f} | std:{:.4f} \n".format(model_path, R_mean, R_std))

                        cont = False

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == "":
            print("Error: the path for morels is not found!")
        else:
            # Build and load the generator
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print("Load text encoder from:", cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            device_ids = [int(cfg.GPU_ID)]
            netG = G_NET()
            netG = _CustomDataParallel(netG, device_ids=device_ids)
            s_tmp = cfg.TRAIN.NET_G[: cfg.TRAIN.NET_G.rfind(".pth")]
            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print("Load G from: ", model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = "%s/%s" % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = captions == 0
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask, cap_lens)
                    # G attention
                    for j in range(batch_size):
                        save_name = "%s/%d_s_%d" % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = "%s_g%d.png" % (save_name, k)
                            im.save(fullpath)
