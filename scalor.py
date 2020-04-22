import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from discovery import ProposalRejectionCell
from propagation import PropagationCell
from modules import ImgEncoder, ZWhatEnc, GlimpseDec, BgDecoder, BgEncoder, ConvLSTMEncoder
from common import *


class SCALOR(nn.Module):

    def __init__(self, args):
        super(SCALOR, self).__init__()
        self.args = args
        self.bg_what_std_bias = 0

        if args.phase_conv_lstm:
            self.image_enc = ConvLSTMEncoder(args)
        else:
            self.image_enc = ImgEncoder(args)

        self.z_what_net = ZWhatEnc()
        self.glimpse_dec_net = GlimpseDec()

        self.propagate_cell = PropagationCell(
                args,
                z_what_net=self.z_what_net,
                glimpse_dec_net=self.glimpse_dec_net
        )
        if not self.args.phase_no_background:
            self.bg_enc = BgEncoder()
            self.bg_dec = BgDecoder()
            self.bg_prior_rnn = nn.GRUCell(bg_what_dim, bg_prior_rnn_hid_dim)
            self.bg_prior_net = nn.Linear(bg_prior_rnn_hid_dim, bg_what_dim * 2)

        self.proposal_rejection_cell = ProposalRejectionCell(
                args,
                z_what_net=self.z_what_net,
                glimpse_dec_net=self.glimpse_dec_net
        )

        if args.phase_parallel:
            self.image_enc = nn.DataParallel(self.image_enc)
            self.propagate_cell = nn.DataParallel(self.propagate_cell)
            self.bg_enc = nn.DataParallel(self.bg_enc)
            self.bg_dec = nn.DataParallel(self.bg_dec)
            self.proposal_rejection_cell = nn.DataParallel(self.proposal_rejection_cell)

        self.register_buffer('z_pres_disc_threshold', torch.tensor(0.7))
        self.register_buffer('prior_bg_mean_t1', torch.zeros(1))
        self.register_buffer('prior_bg_std_t1', torch.ones(1))
        self.register_buffer('color_t', self.args.color_t)

        self.prior_rnn_init_out = None
        self.prior_rnn_init_hid = None

        self.bg_prior_rnn_init_hid = None

    @property
    def p_bg_what_t1(self):
        return Normal(self.prior_bg_mean_t1, self.prior_bg_std_t1)

    def initial_temporal_rnn_hid(self, device):
        return torch.zeros((1, temporal_rnn_out_dim)).to(device), \
               torch.zeros((1, temporal_rnn_hid_dim)).to(device)

    def initial_prior_rnn_hid(self, device):
        if self.prior_rnn_init_out is None or self.prior_rnn_init_hid is None:
            self.prior_rnn_init_out = torch.zeros(1, prior_rnn_out_dim).to(device)
            self.prior_rnn_init_hid = torch.zeros(1, prior_rnn_hid_dim).to(device)

        return self.prior_rnn_init_out, self.prior_rnn_init_hid

    def initial_bg_prior_rnn_hid(self, device):
        if self.bg_prior_rnn_init_hid is None:
            self.bg_prior_rnn_init_hid = torch.zeros(1, bg_prior_rnn_hid_dim).to(device)

        return self.bg_prior_rnn_init_hid

    def forward(self, seq, eps=1e-15):

        bs = seq.size(0)
        seq_len = seq.size(1)
        device = seq.device

        temporal_rnn_out_pre = seq.new_zeros(
            bs, 1, temporal_rnn_out_dim).to(device)
        temporal_rnn_hid_pre = seq.new_zeros(bs, 1, temporal_rnn_hid_dim)
        prior_rnn_out_pre = seq.new_zeros(bs, 1, prior_rnn_out_dim)
        prior_rnn_hid_pre = seq.new_zeros(bs, 1, prior_rnn_hid_dim)
        z_what_pre = seq.new_zeros(bs, 1, z_what_dim)
        z_where_pre = seq.new_zeros(bs, 1, (z_where_scale_dim + z_where_scale_dim))
        z_where_bias_pre = seq.new_zeros(bs, 1, (z_where_scale_dim + z_where_scale_dim))
        z_depth_pre = seq.new_zeros(bs, 1, z_depth_dim)
        z_pres_pre = seq.new_zeros(bs, 1, z_pres_dim)
        cumsum_one_minus_z_pres_prop_pre = seq.new_zeros(bs, 1, z_pres_dim)
        ids_pre = seq.new_zeros(bs, 1)

        lengths = seq.new_zeros(bs)

        kl_z_pres_all = seq.new_zeros(bs, seq_len)
        kl_z_what_all = seq.new_zeros(bs, seq_len)
        kl_z_where_all = seq.new_zeros(bs, seq_len)
        kl_z_depth_all = seq.new_zeros(bs, seq_len)
        kl_z_bg = seq.new_zeros(bs, seq_len)
        log_imp_all = seq.new_zeros(bs, seq_len)
        log_like_all = seq.new_zeros(bs, seq_len)
        y_seq = seq.new_zeros(bs, seq_len, 3, img_h, img_w)

        log_disc_list = []
        log_prop_list = []
        scalor_log_list = []
        counting_list = []

        if self.args.phase_conv_lstm:
            img_enc_seq, lstm_gate_log = self.image_enc(seq)
        else:
            img_enc_seq = self.image_enc(seq.view(-1, seq.size(-3), seq.size(-2), seq.size(-1)))
            img_enc_seq = img_enc_seq.view(bs, -1, img_enc_seq.size(-3), img_enc_seq.size(-2), img_enc_seq.size(-1))

        bg_rnn_hid_pre = self.initial_bg_prior_rnn_hid(device).expand(bs, -1)

        for i in range(seq_len):
            x = seq[:, i]
            kl_z_what_prop = seq.new_zeros(bs)
            kl_z_where_prop = seq.new_zeros(bs)
            kl_z_depth_prop = seq.new_zeros(bs)
            kl_z_pres_prop = seq.new_zeros(bs)
            log_imp_prop = seq.new_zeros(bs)
            log_prop = None

            img_enc = img_enc_seq[:, i]

            if lengths.max() != 0:

                max_length = int(torch.max(lengths))

                y_each_obj_prop, alpha_map_prop, importance_map_prop, z_what_prop, z_where_prop, \
                z_where_bias_prop, z_depth_prop, z_pres_prop, ids_prop, kl_z_what_prop, kl_z_where_prop, \
                kl_z_depth_prop, kl_z_pres_prop, temporal_rnn_out, temporal_rnn_hid, prior_rnn_out, prior_rnn_hid, \
                cumsum_one_minus_z_pres_prop, log_imp_prop, log_prop = \
                    self.propagate_cell(
                        x, img_enc, temporal_rnn_out_pre, temporal_rnn_hid_pre, prior_rnn_out_pre, prior_rnn_hid_pre,
                        z_what_pre, z_where_pre, z_where_bias_pre, z_depth_pre, z_pres_pre,
                        cumsum_one_minus_z_pres_prop_pre, ids_pre, lengths, max_length, i, eps=eps
                    )
            else:
                z_what_prop = x.new_zeros(bs, 1, z_what_dim)
                z_where_prop = x.new_zeros(bs, 1, (z_where_scale_dim + z_where_shift_dim))
                z_where_bias_prop = x.new_zeros(bs, 1, (z_where_scale_dim + z_where_shift_dim))
                z_depth_prop = seq.new_zeros(bs, 1, z_depth_dim)
                z_pres_prop = x.new_zeros(bs, z_pres_dim)
                cumsum_one_minus_z_pres_prop = x.new_zeros(bs, 1, z_pres_dim)
                y_each_obj_prop = x.new_zeros(bs, 1, 3, img_h, img_w)
                alpha_map_prop = x.new_zeros(bs, 1, 1, img_h, img_w)
                importance_map_prop = x.new_zeros(bs, 1, 1, img_h, img_w)
                ids_prop = seq.new_zeros(bs, 1)

            alpha_map_prop_sum = alpha_map_prop.sum(1)
            alpha_map_prop_sum = \
                alpha_map_prop_sum + (alpha_map_prop_sum.clamp(eps, 1 - eps) - alpha_map_prop_sum).detach()
            y_each_obj_disc, alpha_map_disc, importance_map_disc, \
            z_what_disc, z_where_disc, z_where_bias_disc, z_depth_disc, \
            z_pres_disc, ids_disc, kl_z_what_disc, kl_z_where_disc, \
            kl_z_pres_disc, kl_z_depth_disc, log_imp_disc, log_disc = \
                self.proposal_rejection_cell(
                    x, img_enc, alpha_map_prop_sum, ids_prop, lengths, i, eps=eps
                )

            importance_map = torch.cat((importance_map_prop, importance_map_disc), dim=1)

            importance_map_norm = importance_map / (importance_map.sum(dim=1, keepdim=True) + eps)

            # (bs, 1, img_h, img_w)
            alpha_map = torch.cat((alpha_map_prop, alpha_map_disc), dim=1).sum(dim=1)

            alpha_map = alpha_map + (alpha_map.clamp(eps, 1 - eps) - alpha_map).detach()

            y_each_obj = torch.cat((y_each_obj_prop, y_each_obj_disc), dim=1)

            y_nobg = (y_each_obj.view(bs, -1, 3, img_h, img_w) * importance_map_norm).sum(dim=1)

            if not self.args.phase_no_background:
                if i == 0:
                    p_bg_what = self.p_bg_what_t1
                else:
                    bg_rnn_hid_pre = self.bg_prior_rnn(bg_what_pre, bg_rnn_hid_pre)
                    # bg_rnn_hid_pre = self.layer_norm_h(bg_rnn_hid_pre)
                    p_bg_what_mean_bias, p_bg_what_std = self.bg_prior_net(bg_rnn_hid_pre).chunk(2, -1)
                    p_bg_what_mean = p_bg_what_mean_bias + bg_what_pre
                    p_bg_what_std = F.softplus(p_bg_what_std + self.bg_what_std_bias)
                    p_bg_what = Normal(p_bg_what_mean, p_bg_what_std)

                x_alpha_cat = torch.cat((x, (1 - alpha_map)), dim=1)
                # Background
                z_bg_mean, z_bg_std = self.bg_enc(x_alpha_cat)
                z_bg_std = F.softplus(z_bg_std + self.bg_what_std_bias)

                if self.args.phase_generate and i >= self.args.observe_frames:
                    q_bg = p_bg_what
                else:
                    q_bg = Normal(z_bg_mean, z_bg_std)
                z_bg = q_bg.rsample()
                # bg, one_minus_alpha_map = self.bg_dec(z_bg)
                bg = self.bg_dec(z_bg)

                bg_what_pre = z_bg

            else:
                p_bg_what_mean = seq.new_zeros(1)
                p_bg_what_std = seq.new_zeros(1)
                z_bg_mean = seq.new_zeros(1)
                z_bg_std = seq.new_zeros(1)
                bg = seq.new_zeros(bs, 3, img_h, img_w)

            y = y_nobg + (1 - alpha_map) * bg

            p_x_z = Normal(y.flatten(1), self.args.sigma)
            log_like = p_x_z.log_prob(x.view(-1, 3, img_h, img_w).
                                      expand_as(y).flatten(1)).sum(-1)  # sum image dims (C, H, W)

            if not self.args.phase_no_background:
                # Alpha map kl
                kl_z_bg[:, i] = kl_divergence(q_bg, p_bg_what).sum(1)

                ########################################### Compute log importance ############################################
                if not self.training and self.args.phase_nll:
                    # (bs, dim)
                    log_imp_bg = (p_bg_what.log_prob(z_bg) - q_bg.log_prob(z_bg)).sum(1)

                ######################################## End of Compute log importance #########################################

            kl_z_pres_all[:, i] = kl_z_pres_disc + kl_z_pres_prop
            kl_z_what_all[:, i] = kl_z_what_disc + kl_z_what_prop
            kl_z_where_all[:, i] = kl_z_where_disc + kl_z_where_prop
            kl_z_depth_all[:, i] = kl_z_depth_disc + kl_z_depth_prop
            if not self.training and self.args.phase_nll:
                log_imp_all[:, i] = log_imp_disc + log_imp_prop + log_imp_bg
            log_like_all[:, i] = log_like
            y_seq[:, i] = y

            prior_rnn_out_init, prior_rnn_hid_init = self.initial_prior_rnn_hid(device)
            temporal_rnn_out_init, temporal_rnn_hid_init = self.initial_temporal_rnn_hid(device)

            new_prior_rnn_out_init = prior_rnn_out_init.unsqueeze(0). \
                expand((bs, z_what_disc.size(1), prior_rnn_out_dim))
            new_prior_rnn_hid_init = prior_rnn_hid_init.unsqueeze(0). \
                expand((bs, z_what_disc.size(1), prior_rnn_hid_dim))
            new_temporal_rnn_out_init = temporal_rnn_out_init.unsqueeze(0). \
                expand((bs, z_what_disc.size(1), temporal_rnn_out_dim))
            new_temporal_rnn_hid_init = temporal_rnn_hid_init.unsqueeze(0). \
                expand((bs, z_what_disc.size(1), temporal_rnn_hid_dim))

            if lengths.max() != 0:
                z_what_prop_disc = torch.cat((z_what_prop, z_what_disc), dim=1)
                z_where_prop_disc = torch.cat((z_where_prop, z_where_disc), dim=1)
                z_where_bias_prop_disc = torch.cat((z_where_bias_prop, z_where_bias_disc), dim=1)
                z_depth_prop_disc = torch.cat((z_depth_prop, z_depth_disc), dim=1)
                z_pres_prop_disc = torch.cat((z_pres_prop, z_pres_disc), dim=1)
                z_mask_prop_disc = torch.cat((
                    (z_pres_prop > 0).float(),
                    (z_pres_disc > self.z_pres_disc_threshold).float()
                ), dim=1)
                temporal_rnn_out_prop_disc = torch.cat((temporal_rnn_out, new_temporal_rnn_out_init), dim=1)
                temporal_rnn_hid_prop_disc = torch.cat((temporal_rnn_hid, new_temporal_rnn_hid_init), dim=1)
                prior_rnn_out_prop_disc = torch.cat((prior_rnn_out, new_prior_rnn_out_init), dim=1)
                prior_rnn_hid_prop_disc = torch.cat((prior_rnn_hid, new_prior_rnn_hid_init), dim=1)
                cumsum_one_minus_z_pres_prop_disc = torch.cat([cumsum_one_minus_z_pres_prop,
                                                               seq.new_zeros(bs, z_what_disc.size(1), z_pres_dim)],
                                                              dim=1)
                ids_prop_disc = torch.cat((ids_prop, ids_disc), dim=1)
            else:
                z_what_prop_disc = z_what_disc
                z_where_prop_disc = z_where_disc
                z_where_bias_prop_disc = z_where_bias_disc
                z_depth_prop_disc = z_depth_disc
                z_pres_prop_disc = z_pres_disc
                temporal_rnn_out_prop_disc = new_temporal_rnn_out_init
                temporal_rnn_hid_prop_disc = new_temporal_rnn_hid_init
                prior_rnn_out_prop_disc = new_prior_rnn_out_init
                prior_rnn_hid_prop_disc = new_prior_rnn_hid_init
                z_mask_prop_disc = (z_pres_disc > self.z_pres_disc_threshold).float()
                cumsum_one_minus_z_pres_prop_disc = seq.new_zeros(bs, z_what_disc.size(1), z_pres_dim)
                ids_prop_disc = ids_disc

            num_obj_each = torch.sum(z_mask_prop_disc, dim=1)
            max_num_obj = int(torch.max(num_obj_each).item())

            z_what_pre = seq.new_zeros(bs, max_num_obj, z_what_dim)
            z_where_pre = seq.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
            z_where_bias_pre = seq.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
            z_depth_pre = seq.new_zeros(bs, max_num_obj, z_depth_dim)
            z_pres_pre = seq.new_zeros(bs, max_num_obj, z_pres_dim)
            z_mask_pre = seq.new_zeros(bs, max_num_obj, z_pres_dim)
            temporal_rnn_out_pre = seq.new_zeros(bs, max_num_obj, temporal_rnn_out_dim)
            temporal_rnn_hid_pre = seq.new_zeros(bs, max_num_obj, temporal_rnn_hid_dim)
            prior_rnn_out_pre = seq.new_zeros(bs, max_num_obj, prior_rnn_out_dim)
            prior_rnn_hid_pre = seq.new_zeros(bs, max_num_obj, prior_rnn_hid_dim)
            cumsum_one_minus_z_pres_prop_pre = seq.new_zeros(bs, max_num_obj, z_pres_dim)
            ids_pre = seq.new_zeros(bs, max_num_obj)

            for b in range(bs):
                num_obj = int(num_obj_each[b])

                idx = z_mask_prop_disc[b].nonzero()[:, 0]

                z_what_pre[b, :num_obj] = z_what_prop_disc[b, idx]
                z_where_pre[b, :num_obj] = z_where_prop_disc[b, idx]
                z_where_bias_pre[b, :num_obj] = z_where_bias_prop_disc[b, idx]
                z_depth_pre[b, :num_obj] = z_depth_prop_disc[b, idx]
                z_pres_pre[b, :num_obj] = z_pres_prop_disc[b, idx]
                z_mask_pre[b, :num_obj] = z_mask_prop_disc[b, idx]
                temporal_rnn_out_pre[b, :num_obj] = temporal_rnn_out_prop_disc[b, idx]
                temporal_rnn_hid_pre[b, :num_obj] = temporal_rnn_hid_prop_disc[b, idx]
                prior_rnn_out_pre[b, :num_obj] = prior_rnn_out_prop_disc[b, idx]
                prior_rnn_hid_pre[b, :num_obj] = prior_rnn_hid_prop_disc[b, idx]
                cumsum_one_minus_z_pres_prop_pre[b, :num_obj] = cumsum_one_minus_z_pres_prop_disc[b, idx]
                ids_pre[b, :num_obj] = ids_prop_disc[b, idx]

            if not self.args.phase_do_remove_detach or self.args.global_step < self.args.remove_detach_step:
                z_what_pre = z_what_pre.detach()
                z_where_pre = z_where_pre.detach()
                z_depth_pre = z_depth_pre.detach()
                z_pres_pre = z_pres_pre.detach()
                z_mask_pre = z_mask_pre.detach()
                temporal_rnn_out_pre = temporal_rnn_out_pre.detach()
                temporal_rnn_hid_pre = temporal_rnn_hid_pre.detach()
                prior_rnn_out_pre = prior_rnn_out_pre.detach()
                prior_rnn_hid_pre = prior_rnn_hid_pre.detach()
                cumsum_one_minus_z_pres_prop_pre = cumsum_one_minus_z_pres_prop_pre.detach()
                z_where_bias_pre = z_where_bias_pre.detach()

            lengths = torch.sum(z_mask_pre, dim=(1, 2)).view(bs)

            scalor_step_log = {}
            if self.args.log_phase:
                if ids_prop_disc.size(1) < importance_map_norm.size(1):
                    ids_prop_disc = torch.cat((x.new_zeros(ids_prop_disc[:, 0:1].size()), ids_prop_disc), dim=1)
                id_color = self.color_t[ids_prop_disc.view(-1).long() % self.args.color_num]

                # (bs, num_obj_prop + num_cell_h * num_cell_w, 3, 1, 1)
                id_color = id_color.view(bs, -1, 3, 1, 1)
                # (bs, num_obj_prop + num_cell_h * num_cell_w, 3, img_h, img_w)
                id_color_map = (torch.cat((alpha_map_prop, alpha_map_disc), dim=1) > .3).float() * id_color
                mask_color = (id_color_map * importance_map_norm.detach()).sum(dim=1)
                x_mask_color = x - 0.7 * (alpha_map > .3).float() * (x - mask_color)
                scalor_step_log = {
                    'y_each_obj': y_each_obj.cpu().detach(),
                    'importance_map_norm': importance_map_norm.cpu().detach(),
                    'importance_map': importance_map.cpu().detach(),
                    'bg': bg.cpu().detach(),
                    'alpha_map': alpha_map.cpu().detach(),
                    'x_mask_color': x_mask_color.cpu().detach(),
                    'mask_color': mask_color.cpu().detach(),
                    'p_bg_what_mean': p_bg_what_mean.cpu().detach() if i > 0 else self.p_bg_what_t1.mean.cpu().detach(),
                    'p_bg_what_std': p_bg_what_std.cpu().detach() if i > 0 else self.p_bg_what_t1.stddev.cpu().detach(),
                    'z_bg_mean': z_bg_mean.cpu().detach(),
                    'z_bg_std': z_bg_std.cpu().detach()
                }
                if self.args.phase_conv_lstm:
                    for l_n, log in enumerate(lstm_gate_log):
                        for k, v in log[i].items():
                            scalor_step_log[f'lstm_{k}_layer_{l_n}'] = v.cpu().detach()
                if log_disc:
                    for k, v in log_disc.items():
                        log_disc[k] = v.cpu().detach()
                if log_prop:
                    for k, v in log_prop.items():
                        log_prop[k] = v.cpu().detach()

            log_disc_list.append(log_disc)
            log_prop_list.append(log_prop)
            scalor_log_list.append(scalor_step_log)
            counting_list.append(lengths)

        # (bs, seq_len)
        counting = torch.stack(counting_list, dim=1)

        return y_seq, \
               log_like_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_what_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_where_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_depth_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_pres_all.flatten(start_dim=1).mean(dim=1), \
               kl_z_bg.flatten(start_dim=1).mean(dim=1), \
               log_imp_all.flatten(start_dim=1).sum(dim=1), \
               counting, log_disc_list, log_prop_list, scalor_log_list
