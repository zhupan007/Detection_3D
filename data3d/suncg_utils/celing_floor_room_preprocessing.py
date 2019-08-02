from utils3d.bbox3d_ops import Bbox3D
import numpy as np

def preprocess_cfr(bboxes):
  pass

def preprocess_cfr_standard(bboxes):
  #Bbox3D.draw_bboxes(bboxes, 'Z', False)
  n = bboxes.shape[0]
  mdifs = []
  for i in range(1,n):
    difs = []
    for j in range(i):
      difs.append( np.sum(np.abs( bboxes[i] - bboxes[j] )) )
    mdifs.append( min(difs) )
  mdifs = np.array(mdifs)
  keep_mask = mdifs > 0.1
  bboxes = bboxes[keep_mask]
  return bboxes


