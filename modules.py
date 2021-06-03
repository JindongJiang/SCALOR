import torch
from torch import nn
from torch.distributions import RelaxedBernoulli
from torch.distributions.utils import broadcast_all
from common import *


class NumericalRelaxedBernoulli(RelaxedBernoulli):

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)

        out = self.temperature.log() + diff - 2 * diff.exp().log1p()

        return out


class BgDecoder(nn.Module):

    def __init__(self):
        super(BgDecoder, self).__init__()

        if img_w == 128:
            self.dec = nn.Sequential(
                nn.Conv2d(bg_what_dim, 256, 1),
                nn.CELU(),
                nn.GroupNorm(16, 256),

                nn.Conv2d(256, 256 * 4 * 4, 1),
                nn.PixelShuffle(4),
                nn.CELU(),
                nn.GroupNorm(16, 256),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(16, 256),

                nn.Conv2d(256, 128 * 4 * 4, 1),
                nn.PixelShuffle(4),
                nn.CELU(),
                nn.GroupNorm(16, 128),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(16, 128),

                nn.Conv2d(128, 64 * 2 * 2, 1),
                nn.PixelShuffle(2),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, 16 * 4 * 4, 1),
                nn.PixelShuffle(4),
                nn.CELU(),
                nn.GroupNorm(4, 16),
                nn.Conv2d(16, 16, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(4, 16),
            )
        elif img_w == 64:
            self.dec = nn.Sequential(
                nn.Conv2d(bg_what_dim, 256, 1),
                nn.CELU(),
                nn.GroupNorm(16, 256),

                nn.Conv2d(256, 128 * 4 * 4, 1),
                nn.PixelShuffle(4),
                nn.CELU(),
                nn.GroupNorm(16, 128),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(16, 128),

                nn.Conv2d(128, 64 * 4 * 4, 1),
                nn.PixelShuffle(4),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, 16 * 4 * 4, 1),
                nn.PixelShuffle(4),
                nn.CELU(),
                nn.GroupNorm(4, 16),
                nn.Conv2d(16, 16, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(4, 16)
            )

        self.bg_dec = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

    def forward(self, x):
        o = self.dec(x.view(-1, bg_what_dim, 1, 1))
        bg = torch.sigmoid(self.bg_dec(o))
        # alpha_map = torch.sigmoid(self.alpha_dec(o))

        # return bg, alpha_map
        return bg

class BgDecoderSBD(nn.Module):

    def __init__(self):
        super(BgDecoderSBD, self).__init__()

        if img_w == 128:
            self.dec = nn.Sequential(
                nn.Conv2d(bg_what_dim + 2, 64, 5, padding=2),
                nn.CELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, 64, 5, padding=2),
                nn.CELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, 64, 5, padding=2),
                nn.CELU(),
                nn.GroupNorm(8, 64),
            )
        elif img_w == 64:
            self.dec = nn.Sequential(
                nn.Conv2d(bg_what_dim + 2, 64, 5, padding=2),
                nn.CELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, 64, 5, padding=2),
                nn.CELU(),
                nn.GroupNorm(8, 64),
            )

        self.bg_dec = nn.Conv2d(64, 3, 3, padding=1)

        self.register_buffer(
            'position_embedding',
            torch.stack(
                torch.meshgrid(
                    [torch.linspace(-1, 1, img_w), torch.linspace(-1, 1, img_w)]
                ), dim=0
            ).unsqueeze(dim=0)
        )

    def forward(self, x):
        bs = x.size(0)
        inp = torch.cat(
            [
                self.position_embedding.expand(bs, -1, -1, -1),
                x[..., None, None].expand(-1, -1, img_h, img_w)
            ],
            dim=1
        )
        o = self.dec(inp)
        bg = torch.sigmoid(self.bg_dec(o))
        # alpha_map = torch.sigmoid(self.alpha_dec(o))

        # return bg, alpha_map
        return bg


class BgEncoder(nn.Module):

    def __init__(self):
        super(BgEncoder, self).__init__()
        if img_w == 128:
            self.enc = nn.Sequential(
                nn.Conv2d(4, 16, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(4, 16),
                nn.Conv2d(16, 32, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 32),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, bg_what_dim * 2, 4),
            )
        elif img_w == 64:
            self.enc = nn.Sequential(
                nn.Conv2d(4, 16, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(4, 16),
                nn.Conv2d(16, 32, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 32),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, bg_what_dim * 2, 4),
            )

    def forward(self, x):
        bs = x.size(0)
        bg_what_mean, bg_what_std = self.enc(x).view(bs, -1).chunk(2, -1)

        return bg_what_mean, bg_what_std


class ImgEncoder(nn.Module):

    def __init__(self, args):
        super(ImgEncoder, self).__init__()

        self.args = args
        if img_w == 64:
            if self.args.num_cell_h == 8:
                self.enc = nn.Sequential(
                    nn.Conv2d(3, 16, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 16, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 32, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, img_encode_dim, 1),
                    nn.CELU(),
                    nn.GroupNorm(16, img_encode_dim)
                )
            elif self.args.num_cell_h == 4:
                self.enc = nn.Sequential(
                    nn.Conv2d(3, 16, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 16, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 32, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, img_encode_dim, 1),
                    nn.CELU(),
                    nn.GroupNorm(16, img_encode_dim)
                )
        else:
            if self.args.num_cell_h == 8:
                self.enc = nn.Sequential(
                    nn.Conv2d(3, 16, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 16, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 32, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(16, 128),
                    nn.Conv2d(128, img_encode_dim, 1),
                    nn.CELU(),
                    nn.GroupNorm(16, img_encode_dim)
                )
            elif self.args.num_cell_h == 4:
                self.enc = nn.Sequential(
                    nn.Conv2d(3, 16, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 16, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(4, 16),
                    nn.Conv2d(16, 32, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 32),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(16, 128),
                    nn.Conv2d(128, img_encode_dim, 1),
                    nn.CELU(),
                    nn.GroupNorm(16, img_encode_dim)
                )

        self.enc_lat = nn.Sequential(
            nn.Conv2d(img_encode_dim, img_encode_dim, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(img_encode_dim, img_encode_dim, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64)
        )

        self.enc_cat = nn.Sequential(
            nn.Conv2d(img_encode_dim * 2, img_encode_dim, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, img_encode_dim)
        )

    def forward(self, x):
        img_enc = self.enc(x)

        lateral_enc = self.enc_lat(img_enc)

        cat_enc = self.enc_cat(torch.cat((img_enc, lateral_enc), dim=1))

        return cat_enc


class ZWhatEnc(nn.Module):

    def __init__(self):
        super(ZWhatEnc, self).__init__()

        if glimpse_size == 32:
            self.enc_cnn = nn.Sequential(
                nn.Conv2d(3, 16, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(4, 16),
                nn.Conv2d(16, 32, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 32),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, z_what_enc_dim, 4),
                nn.CELU(),
                nn.GroupNorm(16, z_what_enc_dim)
            )
        elif glimpse_size == 64:
            self.enc_cnn = nn.Sequential(
                nn.Conv2d(3, 16, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(4, 16),
                nn.Conv2d(16, 32, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 32),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.CELU(),
                nn.GroupNorm(16, 128),
                nn.Conv2d(128, 128, 4),
                nn.CELU(),
                nn.GroupNorm(16, 128),
            )

        self.enc_what = nn.Linear(128, z_what_dim * 2)

    def forward(self, x):
        x = self.enc_cnn(x)

        z_what_mean, z_what_std = self.enc_what(x.flatten(start_dim=1)).chunk(2, -1)

        return z_what_mean, z_what_std


class GlimpseDec(nn.Module):

    def __init__(self):
        super(GlimpseDec, self).__init__()

        # self.o_bias = -.5
        if glimpse_size == 32:
            self.dec = nn.Sequential(
                nn.Conv2d(z_what_dim, 128, 1),
                nn.CELU(),
                nn.GroupNorm(16, 128),

                nn.Conv2d(128, 64 * 4 * 4, 1),
                nn.PixelShuffle(4),
                nn.CELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, 64 * 2 * 2, 1),
                nn.PixelShuffle(2),
                nn.CELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, 32 * 2 * 2, 1),
                nn.PixelShuffle(2),
                nn.CELU(),
                nn.GroupNorm(8, 32),

                nn.Conv2d(32, 16 * 2 * 2, 1),
                nn.PixelShuffle(2),
                nn.CELU(),
                nn.GroupNorm(4, 16),

                nn.Conv2d(16, 16, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(4, 16),
            )
        elif glimpse_size == 64:
            self.dec = nn.Sequential(
                nn.Conv2d(z_what_dim, 128, 1),
                nn.CELU(),
                nn.GroupNorm(16, 128),

                nn.Conv2d(128, 64 * 4 * 4, 1),
                nn.PixelShuffle(4),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, 64 * 4 * 4, 1),
                nn.PixelShuffle(4),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, 32 * 2 * 2, 1),
                nn.PixelShuffle(2),
                nn.CELU(),
                nn.GroupNorm(8, 32),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(8, 32),

                nn.Conv2d(32, 16 * 2 * 2, 1),
                nn.PixelShuffle(2),
                nn.CELU(),
                nn.GroupNorm(4, 16),

            )
        if img_w == 64:
            self.dec_o = nn.Conv2d(16, 3, 3, 1, 1)
            self.dec_alpha = nn.Conv2d(16, 1, 3, 1, 1)
        else:
            self.dec_o = nn.Sequential(
                nn.Conv2d(16, 8, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(4, 8),
                nn.Conv2d(8, 3, 3, 1, 1)
            )

            self.dec_alpha = nn.Sequential(
                nn.Conv2d(16, 8, 3, 1, 1),
                nn.CELU(),
                nn.GroupNorm(4, 8),
                nn.Conv2d(8, 1, 3, 1, 1)
            )

    def forward(self, x):
        x = self.dec(x.view(x.size(0), -1, 1, 1))

        o = torch.sigmoid(self.dec_o(x))
        alpha = torch.sigmoid(self.dec_alpha(x))

        return o, alpha


class ConvLSTMEncoder(nn.Module):

    def __init__(self, args):
        super(ConvLSTMEncoder, self).__init__()

        self.args = args

        self.image_enc = ImgEncoder(args)

        self.conv_lstm = ConvLSTM(args, input_dim=img_encode_dim, hidden_dim=[conv_lstm_hid_dim, ] * 2,
                                  kernel_size=(3, 3), num_layers=2, map_h=self.args.num_cell_h,
                                  map_w=self.args.num_cell_w)

    def forward(self, x):
        """

        :param x: (bs, T, dim, cell_h, cell_w)
        :return:
        """
        bs = x.size(0)
        img_conv_enc = self.image_enc(x.view(-1, x.size(2), x.size(3), x.size(4)))

        img_conv_enc = img_conv_enc.view(bs, -1, img_conv_enc.size(-3), img_conv_enc.size(-2), img_conv_enc.size(-1))

        img_enc, log_list = self.conv_lstm(img_conv_enc)

        return img_enc[0], log_list


class ConvLSTMCell(nn.Module):

    def __init__(self, args, input_dim, hidden_dim, kernel_size=3, map_h=8, map_w=8):
        super(ConvLSTMCell, self).__init__()

        self.args = args

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv_x = nn.Conv2d(in_channels=self.input_dim,
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=True)

        self.conv_h = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False)

        self.Wci = nn.Parameter(torch.zeros(1, self.hidden_dim, map_h, map_w),
                                requires_grad=True)
        self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_dim, map_h, map_w),
                                requires_grad=True)
        self.Wco = nn.Parameter(torch.zeros(1, self.hidden_dim, map_h, map_w),
                                requires_grad=True)

        self.register_buffer('h_0', torch.zeros(1, self.hidden_dim, 1, 1))
        self.register_buffer('c_0', torch.zeros(1, self.hidden_dim, 1, 1))

    def forward(self, x, h_c):
        h_cur, c_cur = h_c

        xi, xf, xo, xc = self.conv_x(x).split(self.hidden_dim, dim=1)

        hi, hf, ho, hc = self.conv_h(h_cur).split(self.hidden_dim, dim=1)

        i = torch.sigmoid(xi + hi + c_cur * self.Wci)
        f = torch.sigmoid(xf + hf + c_cur * self.Wcf)
        c_next = f * c_cur + i * torch.tanh(xc + hc)
        o = torch.sigmoid(xo + ho + c_cur * self.Wco)
        h_next = o * torch.tanh(c_next)
        log = None
        if self.args.log_phase:
            log = {
                'i': i,
                'f': f,
                'o': o
            }

        return h_next, c_next, log

    def init_hidden(self, batch_size, inp_size):
        return self.h_0.expand(batch_size, -1, inp_size[-2], inp_size[-1]), \
               self.c_0.expand(batch_size, -1, inp_size[-2], inp_size[-1])


class ConvLSTM(nn.Module):

    def __init__(self, args, input_dim, hidden_dim, kernel_size, num_layers, map_h, map_w,
                 batch_first=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.args = args

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(args=args, input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          map_h=map_h, map_w=map_w))

        self.cell_list = nn.ModuleList(cell_list)

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def forward(self, x):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            x = x.permute(1, 0, 2, 3, 4)

        bs = x.size(0)
        layer_output_h_list = []
        layer_log_list = []

        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):

            output_h = []
            log_list = []
            for t in range(seq_len):
                if t == 0:
                    h, c = self.cell_list[layer_idx].init_hidden(bs, x.size())
                h, c, log = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :],
                                                      h_c=[h, c])
                output_h.append(h)

                log_list.append(log)

            layer_output_h = torch.stack(output_h, dim=1)
            cur_layer_input = layer_output_h

            layer_output_h_list.append(layer_output_h)

            layer_log_list.append(log_list)

        if not self.return_all_layers:
            layer_output_h_list = layer_output_h_list[-1:]

        return layer_output_h_list, layer_log_list
