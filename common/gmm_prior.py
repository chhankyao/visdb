# For the GMM prior, we follow the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py

import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from config import cfg


class MaxMixturePrior(nn.Module):
    def __init__(self, model_path, num_gaussians=8, epsilon=1e-16, use_merged=True):
        super(MaxMixturePrior, self).__init__()
        dtype = torch.float32
        np_dtype = np.float32
        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        with open(model_path, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        means = gmm['means'].astype(np_dtype)
        covs = gmm['covars'].astype(np_dtype)
        weights = gmm['weights'].astype(np_dtype)
        
        self.register_buffer('means', torch.tensor(means, dtype=dtype))
        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions)
        self.register_buffer('precisions', torch.tensor(precisions, dtype=dtype))

        sqrdets = np.array([(np.sqrt(np.linalg.det(c))) for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const * (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term', torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov) + epsilon) for cov in covs]
        self.register_buffer('cov_dets', torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def merged_log_likelihood(self, pose):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means
        prec_diff_prod = torch.einsum('mij,bmj->bmi', [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)
        curr_loglikelihood = 0.5 * diff_prec_quadratic - torch.log(self.nll_weights)
        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose):
        likelihoods = []
        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean
            curr_loglikelihood = torch.einsum('bj,ji->bi', [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b', [curr_loglikelihood, diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term + self.random_var_dim * self.pi_term)
            likelihoods.append(curr_loglikelihood)
        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)
        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose):
        if self.use_merged:
            return self.merged_log_likelihood(pose[:,3:])
        else:
            return self.log_likelihood(pose[:,3:])
        
        
gmm = MaxMixturePrior(cfg.gmm_path)
