from __future__ import print_function
import torch
from torch.autograd import Variable
import time
from torch import nn

def normalize1(x, axis=-1):
  """Normalizing to unit length along the specified dimension.
  Args:
    x: pytorch Variable
  Returns:
    x: pytorch Variable, same shape as input
  """
  y = Variable(torch.zeros(x.size()))
  y = y.cuda()
  for i in range(0, x.size(0)):
    temp = x[i,::]
    temp = temp.t()
    temp = 1. * temp / (torch.norm(temp, 2, axis, keepdim=True).expand_as(temp) + 1e-12)
    y[i,::] = temp.t()
  return y

def normalize(x, axis=-1):
  """Normalizing to unit length along the specified dimension.
  Args:
    x: pytorch Variable
  Returns:
    x: pytorch Variable, same shape as input      
  """
  x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
  return x


def euclidean_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  m, n = x.size(0), y.size(0)
  xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
  yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
  dist = xx + yy
  dist.addmm_(1, -2, x, y.t())
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist


def dsr_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  #start = time.time()
  m, n = x.size(0), y.size(0)
  kappa = 0.001
  dist = Variable(torch.zeros(m,n))
  dist = dist.cuda()
  T = kappa * Variable(torch.eye(39))
  T = T.cuda()
  T.detach()

  for i in range(0, m):
    Proj_M = torch.matmul(torch.inverse(torch.matmul(x[i,::].t(), x[i,::])+T), x[i,::].t())
    Proj_M.detach()
    for j in range(0, n):
      w = torch.matmul(Proj_M, y[j,::])
      w.detach()
      a = torch.matmul(x[i,::], w) - y[j,::]
      dist[i,j] = torch.pow(a,2).sum(0).sqrt().mean()
  return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
  """For each anchor, find the hardest positive and negative sample.
  Args:
    dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
    labels: pytorch LongTensor, with shape [N]
    return_inds: whether to return the indices. Save time if `False`(?)
  Returns:
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    p_inds: pytorch LongTensor, with shape [N]; 
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
  NOTE: Only consider the case in which all labels have same num of samples, 
    thus we can cope with all anchors in parallel.
  """

  assert len(dist_mat.size()) == 2
  assert dist_mat.size(0) == dist_mat.size(1)
  N = dist_mat.size(0)

  # shape [N, N]
  is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
  is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

  # `dist_ap` means distance(anchor, positive)
  # both `dist_ap` and `relative_p_inds` with shape [N, 1]
  dist_ap, relative_p_inds = torch.max(
    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
  # `dist_an` means distance(anchor, negative)
  # both `dist_an` and `relative_n_inds` with shape [N, 1]
  dist_an, relative_n_inds = torch.min(
    dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
  # shape [N]
  dist_ap = dist_ap.squeeze(1)
  dist_an = dist_an.squeeze(1)

  if return_inds:
    # shape [N, N]
    ind = (labels.new().resize_as_(labels)
           .copy_(torch.arange(0, N).long())
           .unsqueeze( 0).expand(N, N))
    # shape [N, 1]
    p_inds = torch.gather(
      ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    n_inds = torch.gather(
      ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    # shape [N]
    p_inds = p_inds.squeeze(1)
    n_inds = n_inds.squeeze(1)
    return dist_ap, dist_an, p_inds, n_inds

  return dist_ap, dist_an



def DSR_L(x, y, p_inds, n_inds):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  #start = time.time()
  m = y.size(0)

  kappa = 0.001
  dist_p = Variable(torch.zeros(m, 1))
  dist_n = Variable(torch.zeros(m, 1))
  dist_p = dist_p.cuda()
  dist_n = dist_n.cuda()
  T = kappa * Variable(torch.eye(71))
  T = T.cuda()
  T.detach()

  for i in range(0, m):
    Proj_M1 = torch.matmul(torch.inverse(torch.matmul(x[p_inds[i],:,:].t(), x[p_inds[i],:,:])+T), x[p_inds[i],:,:].t())
    Proj_M1.detach()

    Proj_M2 = torch.matmul(torch.inverse(torch.matmul(x[n_inds[i],:,:].t(), x[n_inds[i],:,:])+T), x[n_inds[i],:,:].t())
    Proj_M2.detach()
    w1 = torch.matmul(Proj_M1, y[i,::])
    w1.detach()
    w2 = torch.matmul(Proj_M2, y[i,::])
    w2.detach()
    a1 = torch.matmul(x[p_inds[i],:,:], w1) - y[i,::]
    a2 = torch.matmul(x[n_inds[i], :, :], w2) - y[i, ::]
    dist_p[i, 0] = torch.pow(a1,2).sum(0).sqrt().mean()
    dist_n[i, 0] = torch.pow(a2, 2).sum(0).sqrt().mean()

  dist_n = dist_n.squeeze(1)
  dist_p = dist_p.squeeze(1)
  return dist_n, dist_p

def global_loss(tri_loss, global_feat, global_feat1, labels, normalize_feature=True, spatial_train):
  """
  Args:
    tri_loss: a `TripletLoss` object
    global_feat: pytorch Variable, shape [N, C]
    labels: pytorch LongTensor, with shape [N]
    normalize_feature: whether to normalize feature to unit length along the 
      Channel dimension
  Returns:
    loss: pytorch Variable, with shape [1]
    p_inds: pytorch LongTensor, with shape [N]; 
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    ==================
    For Debugging, etc
    ==================
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
  """
  if normalize_feature:
    global_feat = normalize(global_feat, axis=-1)
  ##global_feat1 = normalize1(global_feat1, axis=-1)
  # shape [N, N]
  dist_mat = euclidean_dist(global_feat, global_feat)
  #dist_mat1 = dsr_dist(global_feat1, global_feat1)
  #dist_mat2 = dist_mat + dist_mat1
  dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
    dist_mat, labels, return_inds=True)
  if spatial_train:
    dist_n, dist_p = DSR_L(global_feat1, global_feat1, p_inds, n_inds)
    dist_ap=dist_p+dist_ap
    dist_an=dist_an+dist_n
  loss = tri_loss(dist_ap, dist_an)
  return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
