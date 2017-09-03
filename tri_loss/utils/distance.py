"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np
import torch
from torch.autograd import Variable


def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)

def normalize1(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""

  for i in range(0, len(nparray)):
    temp = nparray[i,::].T
    temp = temp / (np.linalg.norm(temp, ord=order, axis=axis, keepdims=True) + np.finfo(np.float32).eps)
    nparray[i,::] = temp.T
  return nparray#/ (norm + np.finfo(np.float32).eps)

def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist

def dsr_dist(array1, array2):
  """ Compute the sptial feature reconstruction of all pairs
   array: [M, N, C] M: the number of query, N: the number of spatial feature, C: the dimension of each spatial feature
   array2: [M, N, C] M: the number of gallery
  :return:
  numpy array with shape [m1, m2]
  """

  
  kappa = 0.001
  dist = torch.zeros(len(array1), len(array2))
  dist = dist.cuda()

  for i in range(0, len(array2)):
    if (i%100==0):
      print('{}/{} batches done'.format(i, len(array2)))
    y = torch.FloatTensor(array2[i])
    y = y.cuda()
    T = kappa * torch.eye(y.size(1))
    T = T.cuda()
    Proj_M = torch.matmul(torch.inverse(torch.matmul(y.t(), y) + T), y.t()) # (Y^{T} * Y + kappa * I)^{-1} * Y^{T}
    for j in range(0, len(array1)):
      temp = array1[j]
      temp = torch.FloatTensor(temp)
      temp = temp.cuda()
      a = torch.matmul(y, torch.matmul(Proj_M, temp)) - temp
      dist[j, i] = torch.pow(a, 2).sum(0).sqrt().mean()
  dist = dist.cpu()
  dist = dist.numpy()
  return dist


