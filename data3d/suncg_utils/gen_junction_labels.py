# 4 Oct 2019 xyz

from utils3d.bbox3d_ops import Bbox3D
import torch, os, glob
import numpy as np

PATH = '/DS/SUNCG/suncg_v1_torch_splited/houses'

def cal_connection_label(corners, all_corners, thres=0.1):
  dif = corners.reshape([-1,1,3]) - all_corners.reshape([1,-1,3])
  dis = np.linalg.norm(dif, axis=2)
  connect_num = np.sum(dis < thres, axis=1) - 1
  return connect_num.reshape(-1,4)

def gen_junction_labels(pth_fn):
  new_pth_fn = pth_fn.replace('houses', 'houses_with_cn')
  if os.path.exists(new_pth_fn):
    print(f'already exist {new_pth_fn}')
    return
    pcl, bboxes0, connect_nums = torch.load(new_pth_fn)
  pcl, bboxes0 = torch.load(pth_fn)
  corners = {}
  all_corners = None
  for obj in bboxes0:
    corners_8 = Bbox3D.bboxes_corners(bboxes0[obj], 'Z', False)
    if corners_8.shape[0] == 0:
      corners[obj] = corners_8
      continue

    corners[obj] = (corners_8[:,Bbox3D._yneg_vs,:] + corners_8[:,Bbox3D._ypos_vs,:])/2.0
    if all_corners is None:
      all_corners = corners[obj].reshape([-1,3])
    else:
      all_corners = np.concatenate([all_corners, corners[obj].reshape([-1,3])],0)
    #Bbox3D.draw_bboxes(bboxes0[obj], 'Z', False)
    #Bbox3D.draw_points(corners[obj].reshape([-1,3]))
    #Bbox3D.draw_points_bboxes(corners[obj].reshape([-1,3]), bboxes0[obj], 'Z', False)

  connect_nums = {}
  for obj in corners:
    connect_nums[obj] = cal_connection_label(corners[obj], all_corners)


  if not os.path.exists( os.path.dirname(new_pth_fn) ):
    os.mkdir( os.path.dirname(new_pth_fn) )
  torch.save([pcl,bboxes0,connect_nums], new_pth_fn)
  print(f'save {new_pth_fn}')

if __name__ == '__main__':
    houses = os.listdir(PATH)
    for h in houses:
      hd = os.path.join(PATH, h)
      fs = glob.glob(hd+'/*.pth')
      for f in fs:
        gen_junction_labels(f)
    pass
