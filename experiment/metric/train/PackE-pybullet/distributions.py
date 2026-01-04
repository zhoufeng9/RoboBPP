import torch

"""
Modify standard PyTorch distributions so they are compaible with this code.
"""

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

class FixedNormal(torch.distributions.Normal):
    def log_prob(self, actions):
        log_prob = super().log_prob(actions)
        return log_prob.sum(-1, keepdim=True)

    def entropy(self):
        entropy = super().entropy()
        return entropy.sum(-1)

    def mode(self):
        return self.mean

