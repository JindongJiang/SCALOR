import numpy as np
import torch
from torch.distributions import Normal, kl_divergence, Bernoulli, RelaxedBernoulli
from torch import nn
import torch.nn.functional as F
from utils import spatial_transform, calc_kl_z_pres_bernoulli
from common import *
from modules import NumericalRelaxedBernoulli


class TrackerRNN(nn.Module):

    def __init__(self, hid_dim):
        super(TrackerRNN, self).__init__()

        self.cell = nn.GRUCell(temporal_rnn_inp_dim, hid_dim)

    def forward(self, h_pre, c_pre, temporal_rnn_inp):
        h = self.cell(temporal_rnn_inp, c_pre)

        # output and hidden, for vanilla rnn, output == hidden
        return h, h


class AttEncoder(nn.Module):

    def __init__(self, args):
        self.args = args
        super(AttEncoder, self).__init__()

        self.temporal_img_conv_net = nn.Sequential(
            nn.Conv2d(img_encode_dim, temporal_img_enc_hid_dim, 1),
            nn.CELU(),
            nn.GroupNorm(8, temporal_img_enc_hid_dim)
        )

        self.temporal_img_enc_net = nn.Linear(
            temporal_img_enc_hid_dim * self.args.num_cell_h // 2 * self.args.num_cell_w // 2, temporal_img_enc_dim)

    def forward(self, img_enc):
        """

        :param x: (bs, dim, img_h, img_w)
        """
        bs = img_enc.size(0)
        x = self.temporal_img_enc_net(self.temporal_img_conv_net(img_enc).view(bs, -1))

        # bs, dim
        return x


class PropagationCell(nn.Module):

    def __init__(self, args, z_what_net, glimpse_dec_net):
        super(PropagationCell, self).__init__()
        self.args = args

        self.z_pres_logits_bias = 2.
        self.where_update_scale = where_update_scale
        self.z_where_std_bias = -2

        # self.z_what_gate_bias = 2
        self.register_buffer('z_pres_stop_threshold', torch.tensor(0.6))

        z_where_transit_bias_net_input_dim = temporal_rnn_out_dim + z_what_dim + z_where_scale_dim + \
                                             z_where_shift_dim + z_where_bias_dim + temporal_img_enc_dim

        self.z_where_transit_bias_net = nn.Sequential(
            nn.Linear(z_where_transit_bias_net_input_dim, z_where_transit_bias_net_hid_dim),
            nn.CELU(),
            nn.Linear(z_where_transit_bias_net_hid_dim, (z_where_scale_dim + z_where_shift_dim) * 2)
        )

        z_depth_transit_net_input_dim = temporal_rnn_out_dim + z_what_dim + temporal_img_enc_dim

        self.z_depth_transit_net = nn.Sequential(
            nn.Linear(z_depth_transit_net_input_dim, z_depth_transit_net_hid_dim),
            nn.CELU(),
            nn.Linear(z_depth_transit_net_hid_dim, z_depth_dim * 2)
        )

        self.z_what_from_temporal_net = nn.Sequential(
            nn.Linear(temporal_rnn_out_dim, z_what_from_temporal_hid_dim),
            nn.CELU(),
            nn.Linear(z_what_from_temporal_hid_dim, z_what_dim * 2)
        )

        z_what_gate_net_inp_dim = temporal_rnn_out_dim + temporal_img_enc_dim

        self.z_what_gate_net = nn.Sequential(
            nn.Linear(z_what_gate_net_inp_dim, 64),
            nn.CELU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

        z_pres_transit_input_dim = temporal_rnn_out_dim + z_where_scale_dim + \
                                   z_where_shift_dim + z_where_bias_dim + z_what_dim

        self.z_pres_transit = nn.Sequential(
            nn.Linear(z_pres_transit_input_dim, z_pres_hid_dim),
            nn.CELU(),
            nn.Linear(z_pres_hid_dim, z_pres_dim),
        )

        temporal_rnn_inp_net_inp_dim = z_where_scale_dim + z_where_shift_dim + z_pres_dim + \
                                       z_what_dim + z_where_bias_dim + temporal_img_enc_dim

        self.temporal_rnn_inp_net = nn.Linear(temporal_rnn_inp_net_inp_dim, temporal_rnn_inp_dim)

        self.temporal_rnn = TrackerRNN(temporal_rnn_hid_dim)
        self.attention_encoding = AttEncoder(self.args)

        self.glimpse_dec_net = glimpse_dec_net
        self.z_what_net = z_what_net
        self.prior_cell = PropagatePrior()

    def forward(self, x, img_enc, temporal_rnn_out_pre, temporal_rnn_hid_pre, prior_rnn_out_pre,
                prior_rnn_hid_pre, z_what_pre, z_where_pre, z_where_bias_pre, z_depth_pre, z_pres_pre,
                cumsum_one_minus_z_pres, ids_pre, lengths, max_length, t, eps=1e-15):
        """

        :param x: input image (bs, c, h, w)
        :param img_enc: input image encode (bs, c, num_cell_h, num_cell_w)
        :param temporal_rnn_out_pre: (bs, max_num_obj, dim)
        :param temporal_rnn_hid_pre: (bs, max_num_obj, dim)
        :param z_what_pre: (bs, max_num_obj, dim)
        :param z_where_pre: (bs, max_num_obj, dim)
        :param z_depth_pre: (bs, max_num_obj, dim)
        :param z_pres_pre: (bs, max_num_obj, dim)
        :param cumsum_one_minus_z_pres: (bs, max_num_obj, dim)
        :param lengths: (bs)
        :return:
        """

        bs = x.size(0)
        device = x.device
        max_num_obj = max_length
        bns = bs * max_num_obj
        obj_mask = (z_pres_pre.view(bs, max_num_obj) != 0).float()

        temporal_rnn_out_pre, temporal_rnn_hid_pre, prior_rnn_out_pre, \
        prior_rnn_hid_pre, z_what_pre, z_where_pre, z_where_bias_pre, z_depth_pre, \
        z_pres_pre, cumsum_one_minus_z_pres = \
            temporal_rnn_out_pre.view(bns, -1), temporal_rnn_hid_pre.view(bns, -1), \
            prior_rnn_out_pre.view(bns, -1), prior_rnn_hid_pre.view(bns, -1), \
            z_what_pre.view(bns, -1), z_where_pre.view(bns, -1), z_where_bias_pre.view(bns, -1), \
            z_depth_pre.view(bns, -1), z_pres_pre.view(bns, -1), \
            cumsum_one_minus_z_pres.view(bns, -1)

        prior_rnn_out, prior_rnn_hid, prior_what_mean, prior_what_std, prior_where_bias_mean, \
        prior_where_bias_std, prior_depth_mean, prior_depth_std, prior_pres_prob = \
            self.prior_cell(prior_rnn_out_pre, prior_rnn_hid_pre, z_what_pre,
                            z_where_pre, z_where_bias_pre, z_depth_pre, z_pres_pre)

        z_where_att = x.new_ones(z_where_pre.size()) * .5
        z_where_att[:, 2:] = z_where_pre[:, 2:].detach()
        img_enc_att = spatial_transform(
            img_enc.unsqueeze(1).expand(-1, max_num_obj, -1, -1, -1).contiguous().
                view(bns, img_encode_dim, self.args.num_cell_h, self.args.num_cell_w), z_where_att,
            (bns, img_encode_dim, self.args.num_cell_h // 2, self.args.num_cell_w // 2), inverse=False
        )
        # bns, dim
        temporal_img_enc = self.attention_encoding(img_enc_att).view(-1, temporal_img_enc_dim)
        temporal_img_enc = \
            temporal_img_enc.view(bs, -1, temporal_img_enc_dim).contiguous().view(-1, temporal_img_enc_dim)

        temporal_rnn_inp_net_inp = torch.cat(
            [z_where_pre, z_pres_pre, z_what_pre, z_where_bias_pre, temporal_img_enc],
            dim=1
        )
        temporal_rnn_inp = self.temporal_rnn_inp_net(temporal_rnn_inp_net_inp)
        # bns, dim
        temporal_rnn_out, temporal_rnn_hid = self.temporal_rnn(
            temporal_rnn_out_pre, temporal_rnn_hid_pre, temporal_rnn_inp
        )

        # z_where transition
        z_where_transit_bias_net_inp = torch.cat(
            [temporal_rnn_out, z_what_pre, z_where_pre, z_where_bias_pre, temporal_img_enc], dim=1
        )
        z_where_bias_mean, z_where_bias_std = \
            self.z_where_transit_bias_net(z_where_transit_bias_net_inp).chunk(2, -1)
        z_where_bias_std = F.softplus(z_where_bias_std + self.z_where_std_bias)
        if self.args.phase_generate and t >= self.args.observe_frames:
            z_where_bias_dist = Normal(prior_where_bias_mean, prior_where_bias_std)
        else:
            z_where_bias_dist = Normal(z_where_bias_mean, z_where_bias_std)

        z_where_bias = z_where_bias_dist.rsample()

        z_where_shift = z_where_pre[:, 2:] + self.where_update_scale * z_where_bias[:, 2:].tanh()

        scale, ratio = z_where_bias[:, :2].tanh().chunk(2, 1)
        scale = self.args.size_anc + self.args.var_s * scale  # add bias to let masking do its job
        ratio = self.args.ratio_anc + self.args.var_anc * ratio
        ratio_sqrt = ratio.sqrt()

        z_where = torch.cat((scale / ratio_sqrt, scale * ratio_sqrt, z_where_shift), dim=1)

        # # always within the image
        z_where = torch.cat((z_where[:, :2], z_where[:, 2:].clamp(-1.05, 1.05)), dim=1)

        # get glimpse encode
        x_att = \
            spatial_transform(
                x.unsqueeze(1).expand(-1, max_num_obj, -1, -1, -1).contiguous().view(bns, 3, img_h, img_w), z_where,
                (bns, 3, glimpse_size, glimpse_size), inverse=False
            )

        z_what_from_enc_mean, z_what_from_enc_std = self.z_what_net(
            x_att
        )
        z_what_from_enc_std = F.softplus(z_what_from_enc_std)

        # z_what transit
        z_what_from_temporal_mean, z_what_from_temporal_std = \
            self.z_what_from_temporal_net(temporal_rnn_out).chunk(2, -1)

        z_what_from_temporal_std = F.softplus(z_what_from_temporal_std)

        z_what_gate_net_inp = torch.cat((temporal_rnn_out, temporal_img_enc), dim=1)
        forget_gate, input_gate = self.z_what_gate_net(z_what_gate_net_inp).chunk(2, -1)

        z_what_mean = input_gate * z_what_from_enc_mean + \
                      forget_gate * z_what_from_temporal_mean

        z_what_std = F.softplus(input_gate * z_what_from_enc_std + \
                                forget_gate * z_what_from_temporal_std)

        if self.args.phase_generate and t >= self.args.observe_frames:
            z_what_dist = Normal(prior_what_mean, prior_what_std)
        else:
            z_what_dist = Normal(z_what_mean, z_what_std)

        z_what = z_what_dist.rsample()

        z_depth_transit_net_inp = torch.cat(
            [temporal_rnn_out, z_what, temporal_img_enc],
            dim=1
        )
        z_depth_mean, z_depth_std = self.z_depth_transit_net(z_depth_transit_net_inp).chunk(2, -1)
        z_depth_std = F.softplus(z_depth_std)

        if self.args.phase_generate and t >= self.args.observe_frames:
            z_depth_dist = Normal(prior_depth_mean, prior_depth_std)
        else:
            z_depth_dist = Normal(z_depth_mean, z_depth_std)

        z_depth = z_depth_dist.rsample()

        # z_pres bns, dim
        z_pres_transit_inp = torch.cat(
            [temporal_rnn_out, z_where, z_where_bias, z_what],
            dim=1
        )
        z_pres_logits = pres_logit_factor * torch.tanh(self.z_pres_transit(z_pres_transit_inp) +
                                                       self.z_pres_logits_bias)
        if self.args.phase_generate and t >= self.args.observe_frames:
            q_z_pres = NumericalRelaxedBernoulli(probs=prior_pres_prob, temperature=self.args.tau)
        else:
            q_z_pres = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=self.args.tau)

        # for z_pres, we end up setting this to one during generation
        z_pres_y = q_z_pres.rsample()
        z_pres = torch.sigmoid(z_pres_y)

        cumsum_one_minus_z_pres += (1 - z_pres) * obj_mask.view(bns, 1)
        z_pres = z_pres * (cumsum_one_minus_z_pres < self.z_pres_stop_threshold).float()

        # (bs, dim, glimpse_size, glimpse_size)
        o_att, alpha_att = self.glimpse_dec_net(z_what)

        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)
        y_att = alpha_att_hat * o_att

        # (bs, 3, img_h, img_w)
        y_each_obj = spatial_transform(y_att, z_where, (bns, 3, img_h, img_w), inverse=True)

        # (batch_size_t, 1, glimpse_size, glimpse_size)
        importance_map = alpha_att_hat * torch.sigmoid(-z_depth).view(-1, 1, 1, 1)

        # (batch_size_t, 1, img_h, img_w)
        importance_map_full_res = spatial_transform(importance_map, z_where, (bns, 1, img_h, img_w),
                                                    inverse=True)

        # (batch_size_t, 1, img_h, img_w)
        alpha_map = spatial_transform(alpha_att_hat, z_where, (bns, 1, img_h, img_w), inverse=True)

        kl_z_pres = \
            (calc_kl_z_pres_bernoulli(z_pres_logits, prior_pres_prob) *
             obj_mask.view(bns)).view(bs, max_num_obj).sum(1)

        prior_what_dist = Normal(prior_what_mean, prior_what_std)
        prior_where_bias_dist = Normal(prior_where_bias_mean, prior_where_bias_std)
        prior_depth_dist = Normal(prior_depth_mean, prior_depth_std)

        kl_z_what = \
            (kl_divergence(z_what_dist, prior_what_dist).sum(1) * \
             z_pres.squeeze() * obj_mask.view(bns)).view(bs, max_num_obj).sum(1)
        kl_z_where = \
            (kl_divergence(z_where_bias_dist, prior_where_bias_dist).sum(1) * \
             z_pres.squeeze() * obj_mask.view(bns)).view(bs, max_num_obj).sum(1)
        kl_z_depth = \
            (kl_divergence(z_depth_dist, prior_depth_dist).sum(1) * \
             z_pres.squeeze() * obj_mask.view(bns)).view(bs, max_num_obj).sum(1)

        ########################################### Compute log importance ############################################
        log_imp = x.new_zeros(bs, 1)
        if not self.training and self.args.phase_nll:
            z_pres_binary = (z_pres > 0.5).float()
            # (bns, dim)
            log_imp_what = (prior_what_dist.log_prob(z_what) - z_what_dist.log_prob(z_what)) * \
                           z_pres_binary * obj_mask.view(bns, 1)
            log_imp_depth = (prior_depth_dist.log_prob(z_depth) - z_depth_dist.log_prob(z_depth)) * \
                            z_pres_binary * obj_mask.view(bns, 1)
            log_imp_where = (prior_where_bias_dist.log_prob(z_where_bias) - z_where_bias_dist.log_prob(z_where_bias)) * \
                            z_pres_binary * obj_mask.view(bns, 1)

            log_pres_prior = z_pres_binary * torch.log(prior_pres_prob + eps) + \
                             (1 - z_pres_binary) * torch.log(1 - prior_pres_prob + eps)

            log_pres_pos = z_pres_binary * torch.log(torch.sigmoid(z_pres_logits) + eps) + \
                           (1 - z_pres_binary) * torch.log(1 - torch.sigmoid(z_pres_logits) + eps)

            log_imp_pres = (log_pres_prior - log_pres_pos) * obj_mask.view(bns, 1)

            log_imp = log_imp_what.view(bs, -1).sum(1) + log_imp_depth.view(bs, -1).sum(1) + \
                      log_imp_where.view(bs, -1).sum(1) + log_imp_pres.view(bs, -1).sum(1)

        ######################################## End of Compute log importance #########################################
        z_what_all = z_what.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)
        z_where_dummy = x.new_ones(bs, max_num_obj, (z_where_scale_dim + z_where_shift_dim)) * .5
        z_where_dummy[:, :, z_where_scale_dim:] = 2
        z_where_all = z_where.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1) + \
                      z_where_dummy * (1 - obj_mask.view(bs, max_num_obj, 1))
        z_where_bias_all = z_where_bias.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)
        z_pres_all = z_pres.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)
        temporal_rnn_hid_all = \
            temporal_rnn_hid.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)
        temporal_rnn_out_all = \
            temporal_rnn_out.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)
        z_depth_all = z_depth.view(bs, max_num_obj, -1) * obj_mask.view(bs, max_num_obj, 1)
        y_each_obj_all = \
            y_each_obj.view(bs, max_num_obj, 3, img_h, img_w) * obj_mask.view(bs, max_num_obj, 1, 1, 1)
        alpha_map_all = \
            alpha_map.view(bs, max_num_obj, 1, img_h, img_w) * obj_mask.view(bs, max_num_obj, 1, 1, 1)
        importance_map_all = \
            importance_map_full_res.view(bs, max_num_obj, 1, img_h, img_w) * \
            obj_mask.view(bs, max_num_obj, 1, 1, 1)

        cumsum_one_minus_z_pres = cumsum_one_minus_z_pres.view(bs, max_num_obj, -1)
        prior_rnn_out = prior_rnn_out.view(bs, max_num_obj, -1)
        prior_rnn_hid = prior_rnn_hid.view(bs, max_num_obj, -1)

        if self.args.log_phase:
            self.log = {
                'z_what': z_what_all,
                'z_where': z_where_all,
                'z_pres': z_pres_all,
                'z_what_std': z_what_std.view(bs, max_num_obj, -1),
                'z_what_mean': z_what_mean.view(bs, max_num_obj, -1),
                'z_where_bias_std': z_where_bias_std.view(bs, max_num_obj, -1),
                'z_where_bias_mean': z_where_bias_mean.view(bs, max_num_obj, -1),
                'glimpse': x_att.view(bs, max_num_obj, 3, glimpse_size, glimpse_size),
                'glimpse_recon': y_att.view(bs, max_num_obj, 3, glimpse_size, glimpse_size),
                'prior_z_pres_prob': prior_pres_prob.view(bs, max_num_obj, -1),
                'prior_where_bias_std': prior_where_bias_std.view(bs, max_num_obj, -1),
                'prior_where_bias_mean': prior_where_bias_mean.view(bs, max_num_obj, -1),
                'prior_what_mean': prior_what_mean.view(bs, max_num_obj, -1),
                'prior_what_std': prior_what_std.view(bs, max_num_obj, -1),
                'lengths': lengths,
                'z_depth': z_depth_all,
                'z_depth_std': z_depth_std.view(bs, max_num_obj, -1),
                'z_depth_mean': z_depth_mean.view(bs, max_num_obj, -1),
                'y_each_obj': y_each_obj_all.view(bs, max_num_obj, 3, img_h, img_w),
                'alpha_map': alpha_map_all.view(bs, max_num_obj, 1, img_h, img_w),
                'importance_map': importance_map_all.view(bs, max_num_obj, 1, img_h, img_w),
                'z_pres_logits': z_pres_logits.view(bs, max_num_obj, -1),
                'z_pres_y': z_pres_y.view(bs, max_num_obj, -1),
                'o_att': o_att.view(bs, max_num_obj, 3, glimpse_size, glimpse_size),
                'z_where_bias': z_where_bias_all,
                'ids': ids_pre
            }
        else:
            self.log = {}

        return y_each_obj_all, alpha_map_all, importance_map_all, z_what_all, z_where_all, \
               z_where_bias_all, z_depth_all, z_pres_all, ids_pre, kl_z_what, kl_z_where, kl_z_depth, \
               kl_z_pres, temporal_rnn_out_all, temporal_rnn_hid_all, prior_rnn_out, \
               prior_rnn_hid, cumsum_one_minus_z_pres, log_imp, self.log


class PropagatePrior(nn.Module):
    """Attention, initial state of rnn is learnable"""

    def __init__(self):
        super(PropagatePrior, self).__init__()

        prior_rnn_inp_net_inp_dim = z_what_dim + z_where_scale_dim + z_where_shift_dim + \
                                    z_where_bias_dim + z_depth_dim + z_pres_dim

        self.prior_rnn_inp_net = nn.Linear(prior_rnn_inp_net_inp_dim, prior_rnn_inp_dim)

        self.prior_rnn = nn.LSTMCell(prior_rnn_inp_dim, prior_rnn_hid_dim)

        self.prior_what_net = nn.Linear(prior_rnn_out_dim, z_what_dim * 2)
        self.prior_where_bias_net = nn.Linear(
            prior_rnn_out_dim, (z_where_scale_dim + z_where_shift_dim) * 2)

        self.prior_depth_net = nn.Linear(prior_rnn_out_dim, z_depth_dim * 2)
        self.prior_pres_net = nn.Linear(prior_rnn_out_dim, z_pres_dim)
        self.prior_z_pres_logits_bias = 5.
        self.where_update_scale = where_update_scale

    def forward(self, prior_rnn_out_pre, prior_rnn_hid_pre, z_what_pre,
                z_where_pre, z_where_bias_pre, z_depth_pre, z_pres_pre, eps=1e-15):
        bns = z_what_pre.size(0)

        z_what_pre_flat = z_what_pre.view(-1, z_what_dim)
        z_where_pre_flat = z_where_pre.view(-1, z_where_scale_dim + z_where_shift_dim)
        z_where_bias_pre_flat = z_where_bias_pre.view(-1, z_where_bias_dim)
        z_depth_pre_flat = z_depth_pre.view(-1, z_depth_dim)
        z_pres_pre_flat = z_pres_pre.view(-1, z_pres_dim)

        prior_rnn_out_pre_flat = prior_rnn_out_pre.view(bns, -1)
        prior_rnn_hid_pre_flat = prior_rnn_hid_pre.view(bns, -1)

        # prior_rnn
        prior_rnn_inp_net_inp = torch.cat((z_what_pre_flat, z_where_pre_flat, z_where_bias_pre_flat,
                                           z_depth_pre_flat, z_pres_pre_flat), dim=1)
        prior_rnn_inp = self.prior_rnn_inp_net(prior_rnn_inp_net_inp)

        prior_rnn_out, prior_rnn_hid = self.prior_rnn(prior_rnn_inp, (prior_rnn_out_pre_flat,
                                                                      prior_rnn_hid_pre_flat))

        prior_what_mean, prior_what_std = self.prior_what_net(prior_rnn_out).chunk(2, -1)
        prior_depth_mean, prior_depth_std = self.prior_depth_net(prior_rnn_out).chunk(2, -1)
        prior_where_bias_mean, prior_where_bias_std = self.prior_where_bias_net(prior_rnn_out).chunk(2, -1)

        prior_pres_probs = torch.sigmoid(self.prior_pres_net(prior_rnn_out) + \
                                         self.prior_z_pres_logits_bias)

        prior_rnn_out = prior_rnn_out.view(bns, -1)
        prior_rnn_hid = prior_rnn_hid.view(bns, -1)
        prior_what_mean = prior_what_mean.view(bns, -1)
        prior_what_std = prior_what_std.view(bns, -1)
        prior_where_bias_mean = prior_where_bias_mean.view(bns, -1)
        prior_where_bias_std = prior_where_bias_std.view(bns, -1)
        prior_depth_mean = prior_depth_mean.view(bns, -1)
        prior_depth_std = prior_depth_std.view(bns, -1)
        prior_pres_probs = prior_pres_probs.view(bns, -1)

        return prior_rnn_out, prior_rnn_hid, prior_what_mean, F.softplus(prior_what_std), \
               prior_where_bias_mean, F.softplus(prior_where_bias_std), prior_depth_mean, \
               F.softplus(prior_depth_std), prior_pres_probs
