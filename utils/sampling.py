import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def get_distance(x):
    """Helper function for margin-based loss. Return a distance matrix given a matrix."""
    n = x.data.shape[0]

    square = (x**2).sum(1).view(-1, 1)
    distance_square = square + square.view(1, -1) - 2.0 * torch.mm(x, x.transpose(1, 0))

    # Adding identity to make sqrt work.
    return torch.sqrt(distance_square+to_var(torch.eye(n))).clamp(0.0, np.inf)

class DistanceWeightedSampling():
    """Distance weighted sampling. See "sampling matters in deep embedding learning"
    paper for details.

    Parameters
    ----------
    batch_k : int
        Number of images per class.

    Inputs:
        - **data**: input tensor with shape (batch_size, embed_dim).
        Here we assume the consecutive batch_k examples are of the same class.
        For example, if batch_k = 5, the first 5 examples belong to the same class,
        6th-10th examples belong to another class, etc.

    Outputs:
        - a_indices: indices of anchors.
        - x[a_indices]: sampled anchor embeddings.
        - x[p_indices]: sampled positive embeddings.
        - x[n_indices]: sampled negative embeddings.
        - x: embeddings of the input batch.
    """
    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4):
        self.batch_k = batch_k
        self.cutoff = cutoff

        # We sample only from negatives that induce a non-zero loss.
        # These are negatives with a distance < nonzero_loss_cutoff.
        # With a margin-based loss, nonzero_loss_cutoff == margin + beta.
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        super(DistanceWeightedSampling, self).__init__()

    def hybrid_forward(self, x):
        k = self.batch_k
        n, d = x.data.shape

        distance = get_distance(x)
        # Cut off to avoid high variance.
        distance = distance.clamp(self.cutoff, np.inf)
        
        # Subtract max(log(distance)) for stability.
        log_weights = ((2.0 - float(d)) * torch.log(distance)
                       - (float(d - 3) / 2) * torch.log(1.0 - 0.25 * (distance ** 2.0)))
        weights = torch.exp(log_weights - torch.max(log_weights))

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = to_var(torch.ones(weights.data.shape))
        for i in range(0, n, k):
            mask[i:i+k, i:i+k] = 0

        # find entrys whose distance < self.nonzero_loss_cutoff
        mask2 = to_var(torch.zeros(weights.data.shape))
        mask2[distance < self.nonzero_loss_cutoff] = 1

        weights = weights * mask * mask2
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.data.cpu().numpy()
        for i in range(n):
            block_idx = i // k

            try:
                n_indices += np.random.choice(n, k-1, p=np_weights[i]).tolist()
            except:
                n_indices += np.random.choice(n, k-1).tolist()
            for j in range(block_idx * k, (block_idx + 1) * k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        a_indices = torch.LongTensor(a_indices).cuda()
        p_indices = torch.LongTensor(p_indices).cuda()
        n_indices = torch.LongTensor(n_indices).cuda()

        return a_indices, torch.index_select(x, 0, a_indices), torch.index_select(x, 0, p_indices), torch.index_select(x, 0, n_indices), x

class MarginLoss():
    """Margin based loss.
    Parameters
    ----------
    margin : float
        Margin between positive and negative pairs.
    nu : float
        Regularization parameter for beta.
    Inputs:
        - anchors: sampled anchor embeddings.
        - positives: sampled positive embeddings.
        - negatives: sampled negative embeddings.
        - beta_in: class-specific betas.
        - a_indices: indices of anchors. Used to get class-specific beta.
    Outputs:
        - Loss.
    """
    def __init__(self, margin=0.2, nu=0.0):
        super(MarginLoss, self).__init__()
        self._margin = margin
        self._nu = nu

    def hybrid_forward(self, anchors, positives, negatives, beta_in, a_indices=None):
        if a_indices is not None:
            # Jointly train class-specific beta.
            beta = beta_in[a_indices]
            beta_reg_loss = torch.sum(beta) * self._nu
        else:
            # Use a constant beta.
            beta = beta_in
            beta_reg_loss = 0.0

        d_ap = torch.sqrt(torch.sum((positives - anchors)**2, dim=1) + 1e-8)
        d_an = torch.sqrt(torch.sum((negatives - anchors)**2, dim=1) + 1e-8)

        pos_loss = (d_ap - beta + self._margin).clamp(0.0, np.inf)
        neg_loss = (beta - d_an + self._margin).clamp(0.0, np.inf)

        pair_cnt = float(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).data.cpu().numpy())

        # Normalize based on the number of pairs.
        loss = (torch.sum(pos_loss + neg_loss) + beta_reg_loss) / pair_cnt
        return loss


d = DistanceWeightedSampling(3)
m = MarginLoss()

a = torch.Tensor(12, 256).uniform_(0, 1)
a = a/torch.norm(a, 2, 1, True)
a = to_var(a)

print(a)
print(get_distance(a))

ai, a, p, n, x = d.hybrid_forward(a)

loss = m.hybrid_forward(a, p, n, 1)
print(loss)