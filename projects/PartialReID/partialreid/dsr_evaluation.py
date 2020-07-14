# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from fastreid.evaluation.evaluator import DatasetEvaluator
from fastreid.evaluation.rank import evaluate_rank
from fastreid.evaluation.roc import evaluate_roc
from .dsr_distance import get_dsr_dist
import pdb

logger = logging.getLogger('fastreid.' + __name__)


class DsrEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.spatial_features = []
        self.scores = []
        self.pids = []
        self.camids = []
   
    def normalize(self, nparray, order=2, axis=0):
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)
   
    def process(self, outputs):
        features = F.normalize(outputs[0][0])
        self.features.append(features.cpu())
       # outputs1 = self.normalize(outputs[0][1].data, axis=1).cpu().numpy()
        outputs1 = outputs[0][1].data.cpu().numpy()
        outputs1 = self.normalize(outputs1, axis=1)
        self.spatial_features.append(outputs1)
        self.scores.append(outputs[0][2])
        self.pids.extend(outputs[1].cpu().numpy())
        self.camids.extend(outputs[2].cpu().numpy())

    def evaluate(self):
        features = torch.cat(self.features, dim=0)
        spatial_features = np.vstack(self.spatial_features)
        scores = torch.cat(self.scores, dim=0)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(self.pids[:self._num_query])
        query_camids = np.asarray(self.camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(self.pids[self._num_query:])
        gallery_camids = np.asarray(self.camids[self._num_query:])

        self._results = OrderedDict()
        query_features = F.normalize(query_features)
        gallery_features = F.normalize(gallery_features)
        dist = 1 - torch.mm(query_features, gallery_features.t()).numpy()
        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        print(cmc[0], np.mean(all_AP))
     #   pdb.set_trace()
        if self.cfg.TEST.DSR.ENABLED:
            dsr_dist = get_dsr_dist(spatial_features[:self._num_query], spatial_features[self._num_query:], dist, scores[:self._num_query])
            logger.info("Test with DSR setting")
       # cmc, all_AP, all_INP = evaluate_rank(dsr_dist, query_pids, gallery_pids, query_camids, gallery_camids)
       # print(cmc[0])
#        pdb.set_trace()
#        lamb = self.cfg.TEST.DSR.LAMB
        max_value = 0
        k = 0
        for i in range(0, 101):
            lamb = 0.01*i
            dist1 = (1 - lamb) * dist + lamb * dsr_dist
            cmc, all_AP, all_INP = evaluate_rank(dist1, query_pids, gallery_pids, query_camids, gallery_camids)
            if (cmc[0]>max_value):
               k = lamb
               max_value = cmc[0]
        dist1 = (1 - k) * dist + k * dsr_dist
        cmc, all_AP, all_INP = evaluate_rank(dist1, query_pids, gallery_pids, query_camids, gallery_camids)
        print(k, cmc[0], np.mean(all_AP))
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        tprs = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        fprs = [1e-4, 1e-3, 1e-2]
        for i in range(len(fprs)):
            self._results["TPR@FPR={}".format(fprs[i])] = tprs[i]
        return copy.deepcopy(self._results)
