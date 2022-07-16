import torch
import torch.nn as nn
from layers import GLU, ResBlock, conv3x3, upBlock
from miscc.config import cfg
from torch.autograd import Variable


# ############## G networks ###################
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, : self.c_dim]
        logvar = x[:, self.c_dim :]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf  # cfg.TEXT.EMBEDDING_DIM

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(nn.Linear(nz, ngf * 4 * 4 * 2, bias=False), nn.BatchNorm1d(ngf * 4 * 4 * 2), GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context_key, content_value):  #
        """
        input: batch x idf x ih x iw (queryL=ihxiw)
        context: batch x idf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context_key.size(0), context_key.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = context_key

        # Get weight
        # (batch x queryL x idf)(batch x idf x sourceL)-->batch x queryL x sourceL
        weight = torch.bmm(targetT, sourceT)

        # --> batch*queryL x sourceL
        weight = weight.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            weight.data.masked_fill_(mask.data, -float("inf"))
        weight = torch.nn.functional.softmax(weight, dim=1)

        # --> batch x queryL x sourceL
        weight = weight.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        weight = torch.transpose(weight, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL) --> batch x idf x queryL
        weightedContext = torch.bmm(content_value, weight)  #
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        weight = weight.view(batch_size, -1, ih, iw)

        return weightedContext, weight


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf, size):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.size = size
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.avg = nn.AvgPool2d(kernel_size=self.size)
        self.A = nn.Linear(self.ef_dim, 1, bias=False)
        self.B = nn.Linear(self.gf_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.M_r = nn.Sequential(nn.Conv1d(ngf, ngf * 2, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.M_w = nn.Sequential(nn.Conv1d(self.ef_dim, ngf * 2, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.key = nn.Sequential(nn.Conv1d(ngf * 2, ngf, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.value = nn.Sequential(nn.Conv1d(ngf * 2, ngf, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.memory_operation = Memory()
        self.response_gate = nn.Sequential(
            nn.Conv2d(self.gf_dim * 2, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid()
        )
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask, cap_lens):
        """
        h_code(image features):  batch x idf x ih x iw (queryL=ihxiw)
        word_embs(word features): batch x cdf x sourceL (sourceL=seq_len)
        c_code: batch x idf x queryL
        att1: batch x sourceL x queryL
        """
        # Memory Writing
        word_embs_T = torch.transpose(word_embs, 1, 2).contiguous()
        h_code_avg = self.avg(h_code).detach()
        h_code_avg = h_code_avg.squeeze(3)
        h_code_avg_T = torch.transpose(h_code_avg, 1, 2).contiguous()
        gate1 = torch.transpose(self.A(word_embs_T), 1, 2).contiguous()
        gate2 = self.B(h_code_avg_T).repeat(1, 1, word_embs.size(2))
        writing_gate = torch.sigmoid(gate1 + gate2)
        h_code_avg = h_code_avg.repeat(1, 1, word_embs.size(2))
        memory = self.M_w(word_embs) * writing_gate + self.M_r(h_code_avg) * (1 - writing_gate)

        # Key Addressing and Value Reading
        key = self.key(memory)
        value = self.value(memory)
        self.memory_operation.applyMask(mask)
        memory_out, att = self.memory_operation(h_code, key, value)

        # Key Response
        response_gate = self.response_gate(torch.cat((h_code, memory_out), 1))
        h_code_new = h_code * (1 - response_gate) + response_gate * memory_out
        h_code_new = torch.cat((h_code_new, h_code_new), 1)

        out_code = self.residual(h_code_new)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super().__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.gf_dim = ngf

        self.ca_net = CA_NET()
        self.fc = nn.Sequential(
            nn.Linear(cfg.GAN.Z_DIM + ncf, ngf * 16 * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 16 * 4 * 4 * 2),
            GLU(),
        )

        # 4 x 4
        self.get_img_4 = GET_IMAGE_G(ngf * 16)

        # 4 x 4  ==> 8 x 8
        self.block_4_to_8 = upBlock(ngf * 16, ngf * 8)
        self.get_img_8 = GET_IMAGE_G(ngf * 8)

        # 8 x 8  ==> 16 x 16
        self.block_8_to_16 = upBlock(ngf * 8, ngf * 4)
        self.get_img_16 = GET_IMAGE_G(ngf * 4)

        # 16 x 16  ==> 32 x 32
        self.block_16_to_32 = upBlock(ngf * 4, ngf * 2)
        self.get_img_32 = GET_IMAGE_G(ngf * 2)

        # 32 x 32  ==> 64 x 64
        self.block_32_to_64 = upBlock(ngf * 2, ngf)
        self.get_img_64 = GET_IMAGE_G(ngf)

        # 64 x 64 ==> 128 x 128
        self.block_64_to_128 = NEXT_STAGE_G(ngf, nef, ncf, 64)
        self.get_img_128 = GET_IMAGE_G(ngf)

        # 128 x 128 ==> 256 x 256
        self.block_128_to_256 = NEXT_STAGE_G(ngf, nef, ncf, 128)
        self.get_img_256 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask, cap_lens):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
        :param word_embs: batch x cdf x seq_len
        :param mask: batch x seq_len
        :return:
        """
        # == Out-skip discriminator ==#
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        c_z_code = torch.cat((c_code, z_code), 1)

        # state size ngf x 4 x 4
        out = self.fc(c_z_code)
        out = out.view(-1, self.gf_dim * 16, 4, 4)
        cur_image = self.get_img_4(out)  # => ngf x 4 x 4
        fake_imgs.append(cur_image)

        out = self.block_4_to_8(out)
        cur_image = self.get_img_8(out)
        fake_imgs.append(cur_image)

        out = self.block_8_to_16(out)
        cur_image = self.get_img_16(out)
        fake_imgs.append(cur_image)

        out = self.block_16_to_32(out)
        cur_image = self.get_img_32(out)
        fake_imgs.append(cur_image)

        out = self.block_32_to_64(out)
        cur_image = self.get_img_64(out)
        fake_imgs.append(cur_image)

        out, attn = self.block_64_to_128(out, c_code, word_embs, mask, cap_lens)
        cur_image = self.get_img_128(out)
        fake_imgs.append(cur_image)
        att_maps.append(attn)

        out, attn = self.block_128_to_256(out, c_code, word_embs, mask, cap_lens)
        cur_image = self.get_img_256(out)
        fake_imgs.append(cur_image)
        att_maps.append(attn)

        return fake_imgs, att_maps, mu, logvar
