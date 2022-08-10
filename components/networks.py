import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

class ProbMLP(nn.Module):
    def __init__(self, cfg, output_dims, input_dims, device,  mu_lower_bound=-1, mu_upper_bound=1):
        super(ProbMLP, self).__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            nn.Linear(input_dims, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
        )
        self.mu = nn.Linear(cfg.hidden_dim, output_dims)  # mean for each state prediction
        self.sigma = nn.Linear(cfg.hidden_dim, output_dims)  # variance for each state prediction
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()

        self.optimizer = optim.Adam(self.parameters(), lr=cfg.alpha)
        self.to(device)
        self.max_logvar = torch.tensor([np.log(0.25)] * output_dims, dtype=float).to(device)
        self.min_logvar = torch.tensor([np.log(0.25)] * output_dims, dtype=float).to(device)
        self.mu_lower_bound = mu_lower_bound
        self.mu_upper_bound = mu_upper_bound

    def forward(self, state):
        hidden = self.model(state)

        # constrain mu to range [mu_lower_bound, mu_upper_bound]
        mu = self.mu(hidden)
        mu = (self.mu_upper_bound - self.mu_lower_bound) * self.sigmoid(mu) + self.mu_lower_bound

        # constrain logvar to upper and lower logvar seen in training data
        logvar = self.sigma(hidden)
        logvar = self.max_logvar - self.softplus1(self.max_logvar - logvar)
        logvar = self.min_logvar + self.softplus2(logvar - self.min_logvar)
        var = torch.exp(logvar).float()

        return mu, var

    def loss(self, mean_pred, var_pred, observation):
        losses = []
        for i, mean in enumerate(mean_pred):
            dist = MultivariateNormal(loc=mean, covariance_matrix=torch.diag(var_pred[i]))
            l = -dist.log_prob(observation[i])
            loss = torch.minimum(l, -torch.log(torch.tensor(1e-12)))
            losses.append(loss)

        return sum(losses)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        if path == None:
            self.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.load_state_dict(torch.load(path, map_location='cpu'))
