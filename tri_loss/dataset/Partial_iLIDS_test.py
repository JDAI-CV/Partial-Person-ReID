from __future__ import print_function
import sys
import time
import os.path as osp
from PIL import Image
import numpy as np
from collections import defaultdict

from .Dataset import Dataset

from ..utils.utils import measure_time
from ..utils.re_ranking import re_ranking
from ..utils.metric import cmc, mean_ap
from ..utils.dataset_utils import parse_im_name
from ..utils.distance import normalize
from ..utils.distance import compute_dist
from ..utils.distance import dsr_dist


class Partial_iLIDS_test(Dataset):
    """
  Args:
    extract_feat_func: a function to extract features. It takes a batch of
      images and returns a batch of features.
    marks: a list, each element e denoting whether the image is from
      query (e == 0), or
      gallery (e == 1), or
      multi query (e == 2) set
  """

    def __init__(
            self,
            im_dir=None,
            im_names=None,
            marks=None,
            extract_feat_func=None,
            separate_camera_set=None,
            single_gallery_shot=None,
            first_match_break=None,
            **kwargs):

        super(Partial_iLIDS_test, self).__init__(dataset_size=len(im_names), **kwargs)

        # The im dir of all images
        self.im_dir = im_dir
        self.im_names = im_names
        self.marks = marks
        self.extract_feat_func = extract_feat_func
        self.separate_camera_set = separate_camera_set
        self.single_gallery_shot = single_gallery_shot
        self.first_match_break = first_match_break

    def set_feat_func(self, extract_feat_func):
        self.extract_feat_func = extract_feat_func

    def get_sample(self, ptr):
        im_name = self.im_names[ptr]
        im_path = osp.join(self.im_dir, im_name)
        im = np.asarray(Image.open(im_path))
        im, _ = self.pre_process_im1(im)
        id = parse_im_name(self.im_names[ptr], 'id')
        cam = parse_im_name(self.im_names[ptr], 'cam')
        # denoting whether the im is from query, gallery, or multi query set
        mark = self.marks[ptr]
        return im, id, cam, im_name, mark

    def next_batch(self):
        if self.epoch_done and self.shuffle:
            self.prng.shuffle(self.im_names)
        samples, self.epoch_done = self.prefetcher.next_batch()
        im_list, ids, cams, im_names, marks = zip(*samples)
        # Transform the list into a numpy array with shape [N, ...]
        ims = np.stack(im_list, axis=0)
        ids = np.array(ids)
        cams = np.array(cams)
        im_names = np.array(im_names)
        marks = np.array(marks)
        return ims, ids, cams, im_names, marks, self.epoch_done

    def extract_feat(self, labels, index):
        """Extract the features of the whole image set.
    Args:
      normalize_feat: True or False, whether to normalize feature to unit length
      verbose: whether to print the progress of extracting feature
    Returns:
      feat: numpy array with shape [N, C]
      ids: numpy array with shape [N]
      cams: numpy array with shape [N]
      im_names: numpy array with shape [N]
      marks: numpy array with shape [N]
    """
        feat, spatial_feat = [], []

        im_dir = '/home/lingxiao.he/Dataset/Partial_iLIDS/'
        for i in range(0, len(labels)):
            im_path = osp.join(im_dir, labels[i])
            im = np.asarray(Image.open(im_path))
            im, _ = self.pre_process_im(im)
            imgs = np.zeros((1, im.ndim, im.shape[1], im.shape[2]))
            imgs[0, :, :, :] = im
            feat_, feat1_ = self.extract_feat_func(imgs)
            feat.append(feat_)
            spatial_feat.append(feat1_)
        return feat, spatial_feat

    def eval(
            self,
            normalize_feat=False,
            to_re_rank=False,
            pool_type='average',
            verbose=True):

        """Evaluate using metric CMC and mAP.
    Args:
      normalize_feat: whether to normalize features before computing distance
      to_re_rank: whether to also report re-ranking scores
      pool_type: 'average' or 'max', only for multi-query case
      verbose: whether to print the intermediate information
    """

        with measure_time('Extracting feature...', verbose=verbose):
            print('Extract probe feature')
            txt_dir = '/home/lingxiao.he/Dataset/Partial_iLIDS/'
            txt_path = osp.join(txt_dir, 'Probe.txt')
            fh = open(txt_path, 'r')
            labels = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                labels.append(words[0])
            Probe, SpatialProbe = self.extract_feat(
                labels, 0)
        Probe = np.vstack(Probe)
        SpatialProbe = np.vstack(SpatialProbe)

        with measure_time('Extracting feature...', verbose=verbose):
            print('Extract gallery feature')
            txt_dir = '/home/lingxiao.he/Dataset/Partial_iLIDS/'
            txt_path = osp.join(txt_dir, 'Gallery.txt')
            fh = open(txt_path, 'r')
            labels = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                labels.append(words[0])
            Gallery, SpatialGallery = self.extract_feat(
                labels, 1)
        Gallery = np.vstack(Gallery)
        SpatialGallery = np.vstack(SpatialGallery)
        # query, gallery, multi-query indices
        query_ids, gallery_ids, query_cams, gallery_cams = [], [], [], []
        for i in range(1, 120):
            query_ids.append(i)
            gallery_ids.append(i)
            query_cams.append(0)
            gallery_cams.append(1)
        query_ids = np.hstack(query_ids)
        query_cams = np.hstack(query_cams)
        gallery_ids = np.hstack(gallery_ids)
        gallery_cams = np.hstack(gallery_cams)

        # A helper function just for avoiding code duplication.
        def compute_score(dist_mat):
            # Compute mean AP
            mAP = mean_ap(
                distmat=dist_mat,
                query_ids=query_ids, gallery_ids=gallery_ids,
                query_cams=query_cams, gallery_cams=gallery_cams)
            # Compute CMC scores
            cmc_scores = cmc(
                distmat=dist_mat,
                query_ids=query_ids, gallery_ids=gallery_ids,
                query_cams=query_cams, gallery_cams=gallery_cams,
                separate_camera_set=self.separate_camera_set,
                single_gallery_shot=self.single_gallery_shot,
                first_match_break=self.first_match_break,
                topk=10)
            return mAP, cmc_scores

        def print_scores(cmc_scores):
            print('[cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
                  .format(*cmc_scores[[0, 4, 9]]))

        ################
        # Single shot #
        ################
        # query-gallery distance
        with measure_time('Computing distance...', verbose=verbose):
            q_g_global_dist = compute_dist(Probe, Gallery, type='euclidean')
            q_g_sppatial_dist = dsr_dist(SpatialProbe, SpatialGallery, q_g_global_dist)
            for lam in range(0, 11):
                mAP1 = []
                cmc_scores1 = []
                weight = lam * 0.1

                # with measure_time('Computing scores...', verbose=verbose):
                q_g_dist = (1 - weight) * q_g_global_dist + weight * q_g_sppatial_dist
                mAP, cmc_scores = compute_score(q_g_dist)
                cmc_scores1.append(cmc_scores)

                print('{:<30}'.format('Single Query:'), end='')
                print_scores(sum(cmc_scores1))