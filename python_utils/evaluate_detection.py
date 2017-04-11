# ---------------------------------------------------------
# Copyright (c) 2015, Saurabh Gupta
# 
# Licensed under The MIT License [see LICENSE for details]
# ---------------------------------------------------------

import utils.cython_bbox
import numpy as np

def inst_bench_image(dt, gt, bOpts, overlap = None):
  
  nDt = len(dt['sc'])
  nGt = len(gt['diff'])
  numInst = np.sum(gt['diff'] == False)


  if overlap == None:
    overlap = utils.cython_bbox.bbox_overlaps(dt['boxInfo'].astype(np.float), gt['boxInfo'].astype(np.float))
  # assert(issorted(-dt.sc), 'Scores are not sorted.\n');
  sc = dt['sc'];

  det    = np.zeros((nGt,1)).astype(np.bool)
  tp     = np.zeros((nDt,1)).astype(np.bool)
  fp     = np.zeros((nDt,1)).astype(np.bool)
  dupDet = np.zeros((nDt,1)).astype(np.bool)
  instId = np.zeros((nDt,1)).astype(np.int32)
  ov     = np.zeros((nDt,1)).astype(np.float32)

  # Walk through the detections in decreasing score
  # and assign tp, fp, fn, tn labels
  for i in xrange(nDt):
    # assign detection to ground truth object if any
    if nGt > 0:
      maxOverlap = overlap[i,:].max(); maxInd = overlap[i,:].argmax();
      instId[i] = maxInd; ov[i] = maxOverlap;
    else:
      maxOverlap = 0; instId[i] = -1; maxInd = -1;
    # assign detection as true positive/don't care/false positive
    if maxOverlap >= bOpts['minoverlap']:
      if gt['diff'][maxInd] == False:
        if det[maxInd] == False:
          # true positive
          tp[i] = True;
          det[maxInd] = True;
        else:
          # false positive (multiple detection)
          fp[i] = True;
          dupDet[i] = True;
    else:
      # false positive
      fp[i] = True;
  return tp, fp, sc, numInst, dupDet, instId, ov


def inst_bench(dt, gt, bOpts, tp = None, fp = None, sc = None, numInst = None):
  """
  ap, rec, prec, npos, details = inst_bench(dt, gt, bOpts, tp = None, fp = None, sc = None, numInst = None)
  dt  - a list with a dict for each image and with following fields
    .boxInfo - info that will be used to cpmpute the overlap with ground truths, a list
    .sc - score 
  gt
    .boxInfo - info used to compute the overlap,  a list 
    .diff - a logical array of size nGtx1, saying if the instance is hard or not
  bOpt
    .minoverlap - the minimum overlap to call it a true positive
  [tp], [fp], [sc], [numInst] 
      Optional arguments, in case the inst_bench_image is being called outside of this function
  """
  if tp is None:
    # We do not have the tp, fp, sc, and numInst, so compute them from the structures gt, and out
    tp = []; fp = []; numInst = []; score = []; dupDet = []; instId = []; ov = []; 
    for i in range(len(gt)):
      # Sort dt by the score
      sc = dt[i]['sc']
      bb = dt[i]['boxInfo']
      ind = np.argsort(sc, axis = 0);
      ind = ind[::-1]
      if len(ind) > 0:
        sc = np.vstack((sc[i,:] for i in ind))
        bb = np.vstack((bb[i,:] for i in ind))
      else: 
        sc = np.zeros((0,1)).astype(np.float)
        bb = np.zeros((0,4)).astype(np.float)
      
      dtI = dict({'boxInfo': bb, 'sc': sc})
      tp_i, fp_i, sc_i, numInst_i, dupDet_i, instId_i, ov_i = inst_bench_image(dtI, gt[i], bOpts)
      tp.append(tp_i); fp.append(fp_i); score.append(sc_i); numInst.append(numInst_i);
      dupDet.append(dupDet_i); instId.append(instId_i); ov.append(ov_i);
  details = {'tp': list(tp), 'fp': list(fp), 'score': list(score), 'dupDet': list(dupDet),  
    'numInst': list(numInst), 'instId': list(instId), 'ov': list(ov)}
  
  tp = np.vstack(tp[:])
  fp = np.vstack(fp[:])
  sc = np.vstack(score[:])

  cat_all = np.hstack((tp,fp,sc))
  ind = np.argsort(cat_all[:,2])
  cat_all = cat_all[ind[::-1],:]
  tp = np.cumsum(cat_all[:,0], axis = 0);
  fp = np.cumsum(cat_all[:,1], axis = 0);
  thresh = cat_all[:,2];
  npos = np.sum(numInst, axis = 0);

  # Compute precision/recall
  rec = tp / npos;
  prec = np.divide(tp, (fp+tp));
  ap = VOCap(rec, prec);
  return ap, rec, prec, npos, details

def VOCap(rec, prec):
  rec = rec.reshape(rec.size,1); prec = prec.reshape(prec.size,1)
  z = np.zeros((1,1)); o = np.ones((1,1));
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])

  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = 0;
  for i in I:
    ap = ap + (mrec[i] - mrec[i-1])*mpre[i];
  return ap
