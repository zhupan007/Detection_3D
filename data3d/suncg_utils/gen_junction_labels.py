# 4 Oct 2019 xyz

from utils3d.bbox3d_ops import Bbox3D
import torch, os, glob

PATH = '/DS/SUNCG/suncg_v1_torch_splited/houses'

def gen_junction_labels(pth_fn):
  pcl, bboxes0 = torch.load(pth_fn)
  corners = {}
  for obj in bboxes0:
    corners_8 = Bbox3D.bboxes_corners(bboxes0[obj], 'Z', False)
    corners[obj] = corners_8[:,Bbox3D._yneg_vs,:] + corners_8[:,Bbox3D._ypos_vs,:]
    corners[obj] = corners_8
    #Bbox3D.draw_bboxes(bboxes0[obj], 'Z', False)
    #Bbox3D.draw_points(corners[obj].reshape([-1,3]))
    Bbox3D.draw_points_bboxes(corners[obj].reshape([-1,3]), bboxes0[obj], 'Z', False)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

if __name__ == '__main__':
    houses = os.listdir(PATH)
    for h in houses:
      hd = os.path.join(PATH, h)
      fs = glob.glob(hd+'/*.pth')
      for f in fs:
        gen_junction_labels(f)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
