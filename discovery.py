import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import probs_to_logits
from torch.distributions import Normal, kl_divergence
from utils import linear_annealing, spatial_transform, calc_kl_z_pres_bernoulli
from modules import NumericalRelaxedBernoulli
from common import *


class ProposalCore(nn.Module):

    def __init__(self, args):
        super(ProposalCore, self).__init__()

        self.args = args
        self.z_pres_bias = 0
        if img_w == 64:
            if self.args.num_cell_h == 8:
                self.mask_enc_net = nn.Sequential(
                    nn.Conv2d(1, 16, 4, 2, 1),
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
                    nn.Conv2d(64, propagate_encode_dim, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, propagate_encode_dim)
                )
            elif self.args.num_cell_h == 4:
                self.mask_enc_net = nn.Sequential(
                    nn.Conv2d(1, 16, 4, 2, 1),
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
                    nn.Conv2d(64, propagate_encode_dim, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, propagate_encode_dim)
                )
        elif img_w == 128:
            if self.args.num_cell_h == 8:
                self.mask_enc_net = nn.Sequential(
                    nn.Conv2d(1, 16, 4, 2, 1),
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
                    nn.Conv2d(64, propagate_encode_dim, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, propagate_encode_dim)
                )
            elif self.args.num_cell_h == 4:
                self.mask_enc_net = nn.Sequential(
                    nn.Conv2d(1, 16, 4, 2, 1),
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
                    nn.Conv2d(64, 64, 4, 2, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, propagate_encode_dim, 1),
                    nn.CELU(),
                    nn.GroupNorm(8, propagate_encode_dim)
                )

        self.img_mask_cat_enc = nn.Sequential(
            nn.Conv2d(img_encode_dim + propagate_encode_dim, img_encode_dim, 1),
            nn.CELU(),
            nn.GroupNorm(16, img_encode_dim),
            nn.Conv2d(img_encode_dim, img_encode_dim, 1),
            nn.CELU(),
            nn.GroupNorm(16, img_encode_dim),
        )
        if img_w == 64:
            self.z_where_net = nn.Conv2d(img_encode_dim, (z_where_shift_dim + z_where_scale_dim) * 2, 1)

            self.z_pres_net = nn.Conv2d(img_encode_dim, z_pres_dim, 1)

            self.z_depth_net = nn.Conv2d(img_encode_dim, z_depth_dim * 2, 1)

        elif img_w == 128:
            self.z_where_net = nn.Sequential(
                nn.Conv2d(img_encode_dim, 64, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, (z_where_shift_dim + z_where_scale_dim) * 2, 1)
            )

            self.z_pres_net = nn.Sequential(
                nn.Conv2d(img_encode_dim, 64, 1),
                nn.CELU(),
                nn.GroupNorm(8, 64),
                nn.Conv2d(64, z_pres_dim, 1)
            )

            self.z_depth_net = nn.Sequential(
                nn.Conv2d(img_encode_dim, 32, 1),
                nn.CELU(),
                nn.GroupNorm(8, 32),
                nn.Conv2d(32, z_depth_dim * 2, 1)
            )
        if self.args.num_cell_h == 8:
            offset_y, offset_x = torch.meshgrid([torch.arange(8.), torch.arange(8.)])
        elif self.args.num_cell_h == 4:
            offset_y, offset_x = torch.meshgrid([torch.arange(4.), torch.arange(4.)])

        self.register_buffer('offset', torch.stack((offset_x, offset_y), dim=0))

    def forward(self, img_enc, alpha, tau, t, gen_pres_probs=None, gen_depth_mean=None,
                gen_depth_std=None, gen_where_mean=None, gen_where_std=None):
        """

        :param x: (bs, dim, img_h, img_w)
        :param propagate_encode: (bs, propagate_encode_dim)
        :param tau:
        :return:
        """

        bs = img_enc.size(0)

        if self.args.phase_generate and t >= self.args.observe_frames:
            gen_pres_logits = probs_to_logits(gen_pres_probs, is_binary=True).view(1, 1, 1, 1). \
                expand(bs, -1, self.args.num_cell_h, self.args.num_cell_w)
            z_pres_logits = gen_pres_logits
            z_depth_mean, z_depth_std = gen_depth_mean.view(1, -1, 1, 1).expand(bs, -1, self.args.num_cell_h,
                                                                                self.args.num_cell_w), \
                                        gen_depth_std.view(1, -1, 1, 1).expand(bs, -1, self.args.num_cell_h,
                                                                               self.args.num_cell_w)
            z_where_mean, z_where_std = gen_where_mean.view(1, -1, 1, 1).expand(bs, -1, self.args.num_cell_h,
                                                                                self.args.num_cell_w), \
                                        gen_where_std.view(1, -1, 1, 1).expand(bs, -1, self.args.num_cell_h,
                                                                               self.args.num_cell_w)
        else:
            mask_enc = self.mask_enc_net(alpha)

            x_alpha_enc = torch.cat((img_enc, mask_enc), dim=1)

            cat_enc = self.img_mask_cat_enc(x_alpha_enc)

            # (bs, 1, 8, 8)
            z_pres_logits = pres_logit_factor * torch.tanh(self.z_pres_net(cat_enc) + self.z_pres_bias)

            # (bs, dim, 8, 8)
            z_depth_mean, z_depth_std = self.z_depth_net(cat_enc).chunk(2, 1)
            z_depth_std = F.softplus(z_depth_std)
            # (bs, 4 + 4, 8, 8)
            z_where_mean, z_where_std = self.z_where_net(cat_enc).chunk(2, 1)
            z_where_std = F.softplus(z_where_std)

        q_z_pres = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=tau)
        z_pres_y = q_z_pres.rsample()

        z_pres = torch.sigmoid(z_pres_y)

        q_z_depth = Normal(z_depth_mean, z_depth_std)

        z_depth = q_z_depth.rsample()

        q_z_where = Normal(z_where_mean, z_where_std)

        z_where = q_z_where.rsample()

        # (bs, dim, 8, 8)
        z_where_origin = z_where.clone()

        scale, ratio = z_where[:, :2].tanh().chunk(2, 1)
        scale = self.args.size_anc + self.args.var_s * scale  # add bias to let masking do its job
        ratio = self.args.ratio_anc + self.args.var_anc * ratio
        ratio_sqrt = ratio.sqrt()
        z_where[:, 0:1] = scale / ratio_sqrt
        z_where[:, 1:2] = scale * ratio_sqrt
        z_where[:, 2:] = 2. / self.args.num_cell_h * (self.offset + 0.5 + z_where[:, 2:].tanh()) - 1

        z_where = z_where.permute(0, 2, 3, 1).reshape(-1, 4)

        return z_where, z_pres, z_depth, z_where_mean, z_where_std, \
               z_depth_mean, z_depth_std, z_pres_logits, z_pres_y, z_where_origin


class ProposalRejectionCell(nn.Module):

    def __init__(self, args, z_what_net, glimpse_dec_net, max_num_obj=15, sigma=0.1):
        super(ProposalRejectionCell, self).__init__()
        self.args = args

        self.z_pres_anneal_start_step = 0000
        self.z_pres_anneal_end_step = 500
        self.z_pres_anneal_start_value = 1e-1
        self.z_pres_anneal_end_value = self.args.z_pres_anneal_end_value
        self.z_pres_masked_prior = 1e-8
        self.likelihood_sigma = sigma
        self.max_num_obj = max_num_obj

        self.ProposalNet = ProposalCore(self.args)
        self.z_what_net = z_what_net
        self.glimpse_dec = glimpse_dec_net

        self.register_buffer('prior_what_mean', torch.zeros(1))
        self.register_buffer('prior_what_std', torch.ones(1))
        self.register_buffer('prior_bg_mean', torch.zeros(1))
        self.register_buffer('prior_bg_std', torch.ones(1))
        self.register_buffer('prior_depth_mean', torch.zeros(1))
        self.register_buffer('prior_depth_std', torch.ones(1))
        self.register_buffer('prior_where_mean',
                             torch.tensor([0., 0., 0., 0.]).view((z_where_scale_dim + z_where_shift_dim), 1, 1))
        self.register_buffer('prior_where_std',
                             torch.tensor([1., 1., 1., 1.]).view((z_where_scale_dim + z_where_shift_dim), 1, 1))
        self.register_buffer('prior_z_pres_prob', torch.tensor(self.z_pres_anneal_start_value))
        self.register_buffer('num_cell', torch.tensor(self.args.num_cell_h * self.args.num_cell_w))

    @property
    def p_bg_what(self):
        return Normal(self.prior_bg_mean, self.prior_bg_std)

    @property
    def p_z_what(self):
        return Normal(self.prior_what_mean, self.prior_what_std)

    @property
    def p_z_depth(self):
        return Normal(self.prior_depth_mean, self.prior_depth_std)

    @property
    def p_z_where(self):
        return Normal(self.prior_where_mean, self.prior_where_std)

    def forward(self, x, img_enc, alpha_map_prop, ids_prop, lengths, t, eps=1e-15):
        """
            :param z_what_prop: (bs, max_num_obj, dim)
            :param z_where_prop: (bs, max_num_obj, 4)
            :param z_pres_prop: (bs, max_num_obj, 1)
            :param alpha_map_prop: (bs, 1, img_h, img_w)
        """
        bs = x.size(0)
        device = x.device
        alpha_map_prop = alpha_map_prop.detach()

        max_num_disc_obj = (self.max_num_obj - lengths).long()

        self.prior_z_pres_prob = linear_annealing(self.args.global_step, self.z_pres_anneal_start_step,
                                                  self.z_pres_anneal_end_step, self.z_pres_anneal_start_value,
                                                  self.z_pres_anneal_end_value, device)

        # z_where: (bs * num_cell_h * num_cell_w, 4)
        # z_pres, z_depth, z_pres_logits: (bs, dim, num_cell_h, num_cell_w)
        z_where, z_pres, z_depth, z_where_mean, z_where_std, \
        z_depth_mean, z_depth_std, z_pres_logits, z_pres_y, z_where_origin = self.ProposalNet(
            img_enc, alpha_map_prop, self.args.tau, t, gen_pres_probs=x.new_ones(1) * self.args.gen_disc_pres_probs,
            gen_depth_mean=self.prior_depth_mean, gen_depth_std=self.prior_depth_std,
            gen_where_mean=self.prior_where_mean, gen_where_std=self.prior_where_std
        )
        num_cell_h, num_cell_w = z_pres.shape[2], z_pres.shape[3]

        q_z_where = Normal(z_where_mean, z_where_std)
        q_z_depth = Normal(z_depth_mean, z_depth_std)

        z_pres_orgin = z_pres

        if self.args.phase_generate and t >= self.args.observe_frames:
            z_what_mean, z_what_std = self.prior_what_mean.view(1, 1).expand(bs * self.args.num_cell_h *
                                                                             self.args.num_cell_w, z_what_dim), \
                                      self.prior_what_std.view(1, 1).expand(bs * self.args.num_cell_h *
                                                                            self.args.num_cell_w, z_what_dim)
            x_att = x.new_zeros(1)
        else:
            # (bs * num_cell_h * num_cell_w, 3, glimpse_size, glimpse_size)
            x_att = spatial_transform(torch.stack(num_cell_h * num_cell_w * (x,), dim=1).view(-1, 3, img_h, img_w),
                                      z_where,
                                      (bs * num_cell_h * num_cell_w, 3, glimpse_size, glimpse_size), inverse=False)

            # (bs * num_cell_h * num_cell_w, dim)
            z_what_mean, z_what_std = self.z_what_net(x_att)
            z_what_std = F.softplus(z_what_std)

        q_z_what = Normal(z_what_mean, z_what_std)

        z_what = q_z_what.rsample()

        # (bs * num_cell_h * num_cell_w, dim, glimpse_size, glimpse_size)
        o_att, alpha_att = self.glimpse_dec(z_what)

        # Rejection
        if phase_rejection and t > 0:
            alpha_map_raw = spatial_transform(alpha_att, z_where, (bs * num_cell_h * num_cell_w, 1, img_h, img_w),
                                              inverse=True)

            alpha_map_proposed = (alpha_map_raw > 0.3).float()

            alpha_map_prop = (alpha_map_prop > 0.1).float().view(bs, 1, 1, img_h, img_w) \
                .expand(-1, num_cell_h * num_cell_w, -1, -1, -1).contiguous().view(-1, 1, img_h, img_w)

            alpha_map_intersect = alpha_map_proposed * alpha_map_prop

            explained_ratio = alpha_map_intersect.view(bs * num_cell_h * num_cell_w, -1).sum(1) / \
                              (alpha_map_proposed.view(bs * num_cell_h * num_cell_w, -1).sum(1) + eps)

            pres_mask = (explained_ratio < self.args.explained_ratio_threshold).view(bs, 1, num_cell_h, num_cell_w).float()

            z_pres = z_pres * pres_mask

        # The following "if" is useful only if you don't have high-memery GPUs, better to remove it if you do
        if self.training and phase_obj_num_contrain:
            z_pres = z_pres.view(bs, -1)

            z_pres_threshold = z_pres.sort(dim=1, descending=True)[0][torch.arange(bs), max_num_disc_obj]

            z_pres_mask = (z_pres > z_pres_threshold.view(bs, -1)).float()

            if self.args.phase_generate and t >= self.args.observe_frames:
                z_pres_mask = x.new_zeros(z_pres_mask.size())

            z_pres = z_pres * z_pres_mask

            z_pres = z_pres.view(bs, 1, num_cell_h, num_cell_w)

        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)

        y_att = alpha_att_hat * o_att

        # (bs * num_cell_h * num_cell_w, 3, img_h, img_w)
        y_each_cell = spatial_transform(y_att, z_where, (bs * num_cell_h * num_cell_w, 3, img_h, img_w),
                                        inverse=True)

        # (bs * num_cell_h * num_cell_w, 1, glimpse_size, glimpse_size)
        importance_map = alpha_att_hat * torch.sigmoid(-z_depth).view(-1, 1, 1, 1)
        # importance_map = -z_depth.view(-1, 1, 1, 1).expand_as(alpha_att_hat)
        # (bs * num_cell_h * num_cell_w, 1, img_h, img_w)
        importance_map_full_res = spatial_transform(importance_map, z_where,
                                                    (bs * num_cell_h * num_cell_w, 1, img_h, img_w),
                                                    inverse=True)

        # (bs * num_cell_h * num_cell_w, 1, img_h, img_w)
        alpha_map = spatial_transform(alpha_att_hat, z_where, (bs * num_cell_h * num_cell_w, 1, img_h, img_w),
                                      inverse=True)

        # (bs * num_cell_h * num_cell_w, z_what_dim)
        kl_z_what = kl_divergence(q_z_what, self.p_z_what) * z_pres_orgin.view(-1, 1)
        # (bs, num_cell_h * num_cell_w, z_what_dim)
        kl_z_what = kl_z_what.view(-1, num_cell_h * num_cell_w, z_what_dim)
        # (bs * num_cell_h * num_cell_w, z_depth_dim)
        kl_z_depth = kl_divergence(q_z_depth, self.p_z_depth) * z_pres_orgin
        # (bs, num_cell_h * num_cell_w, z_depth_dim)
        kl_z_depth = kl_z_depth.view(-1, num_cell_h * num_cell_w, z_depth_dim)
        # (bs, dim, num_cell_h, num_cell_w)
        kl_z_where = kl_divergence(q_z_where, self.p_z_where) * z_pres_orgin
        if phase_rejection and t > 0:
            kl_z_pres = calc_kl_z_pres_bernoulli(z_pres_logits, self.prior_z_pres_prob * pres_mask +
                                                 self.z_pres_masked_prior * (1 - pres_mask))
        else:
            kl_z_pres = calc_kl_z_pres_bernoulli(z_pres_logits, self.prior_z_pres_prob)

        kl_z_pres = kl_z_pres.view(-1, num_cell_h * num_cell_w, z_pres_dim)

        ########################################### Compute log importance ############################################
        log_imp = x.new_zeros(bs, 1)
        if not self.training and self.args.phase_nll:
            z_pres_orgin_binary = (z_pres_orgin > 0.5).float()
            # (bs * num_cell_h * num_cell_w, dim)
            log_imp_what = (self.p_z_what.log_prob(z_what) - q_z_what.log_prob(z_what)) * z_pres_orgin_binary.view(-1,
                                                                                                                   1)
            log_imp_what = log_imp_what.view(-1, num_cell_h * num_cell_w, z_what_dim)
            # (bs, dim, num_cell_h, num_cell_w)
            log_imp_depth = (self.p_z_depth.log_prob(z_depth) - q_z_depth.log_prob(z_depth)) * z_pres_orgin_binary
            # (bs, dim, num_cell_h, num_cell_w)
            log_imp_where = (self.p_z_where.log_prob(z_where_origin) -
                             q_z_where.log_prob(z_where_origin)) * z_pres_orgin_binary
            if phase_rejection and t > 0:
                p_z_pres = self.prior_z_pres_prob * pres_mask + self.z_pres_masked_prior * (1 - pres_mask)
            else:
                p_z_pres = self.prior_z_pres_prob

            z_pres_binary = (z_pres > 0.5).float()

            log_pres_prior = z_pres_binary * torch.log(p_z_pres + eps) + \
                             (1 - z_pres_binary) * torch.log(1 - p_z_pres + eps)

            log_pres_pos = z_pres_binary * torch.log(torch.sigmoid(z_pres_logits) + eps) + \
                           (1 - z_pres_binary) * torch.log(1 - torch.sigmoid(z_pres_logits) + eps)

            log_imp_pres = log_pres_prior - log_pres_pos

            log_imp = log_imp_what.flatten(start_dim=1).sum(dim=1) + log_imp_depth.flatten(start_dim=1).sum(1) + \
                      log_imp_where.flatten(start_dim=1).sum(1) + log_imp_pres.flatten(start_dim=1).sum(1)

        ######################################## End of Compute log importance #########################################

        # (bs, num_cell_h * num_cell_w)
        ids = torch.arange(num_cell_h * num_cell_w).view(1, -1).expand(bs, -1).to(x.device).float() + \
              ids_prop.max(dim=1, keepdim=True)[0] + 1

        if self.args.log_phase:
            self.log = {
                'z_what': z_what,
                'z_where': z_where,
                'z_pres': z_pres,
                'z_pres_logits': z_pres_logits,
                'z_what_std': q_z_what.stddev,
                'z_what_mean': q_z_what.mean,
                'z_where_std': q_z_where.stddev,
                'z_where_mean': q_z_where.mean,
                'x_att': x_att,
                'y_att': y_att,
                'prior_z_pres_prob': self.prior_z_pres_prob.unsqueeze(0),
                'o_att': o_att,
                'alpha_att_hat': alpha_att_hat,
                'alpha_att': alpha_att,
                'y_each_cell': y_each_cell,
                'z_depth': z_depth,
                'z_depth_std': q_z_depth.stddev,
                'z_depth_mean': q_z_depth.mean,
                # 'importance_map_full_res_norm': importance_map_full_res_norm,
                'z_pres_y': z_pres_y,
                'ids': ids
            }
        else:
            self.log = {}

        return y_each_cell.view(bs, num_cell_h * num_cell_w, 3, img_h, img_w), \
               alpha_map.view(bs, num_cell_h * num_cell_w, 1, img_h, img_w), \
               importance_map_full_res.view(bs, num_cell_h * num_cell_w, 1, img_h, img_w), \
               z_what.view(bs, num_cell_h * num_cell_w, -1), z_where.view(bs, num_cell_h * num_cell_w, -1), \
               torch.zeros_like(z_where.view(bs, num_cell_h * num_cell_w, -1)), \
               z_depth.view(bs, num_cell_h * num_cell_w, -1), z_pres.view(bs, num_cell_h * num_cell_w, -1), ids, \
               kl_z_what.flatten(start_dim=1).sum(dim=1), \
               kl_z_where.flatten(start_dim=1).sum(dim=1), \
               kl_z_pres.flatten(start_dim=1).sum(dim=1), \
               kl_z_depth.flatten(start_dim=1).sum(dim=1), \
               log_imp, self.log
