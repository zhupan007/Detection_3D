## April 2019 xyz
import torch, math

def limit_period(val, offset, period):
  '''
    [-pi/2, pi/2]: offset=0.5, period=pi
    [-pi, 0]: offset=1, period=pi
    [0, pi]: offset=-1, period=pi
  '''
  return val - torch.floor(val / period + offset) * period

def angle_dif(val0, val1, aim_scope_id):
    '''
      aim_scope_id 0:[-pi/2, pi/2]
    '''
    dif = val1 - val0
    if aim_scope_id == 0:
      dif = limit_period(dif, 0.5, math.pi)
    else:
      raise NotImplementedError
    return dif


class OBJ_DEF():
  @staticmethod
  def limit_yaw(yaws, yx_zb):
    '''
    standard: [0, pi]
    yx_zb: [-pi/2, pi/2]
    '''
    if yx_zb:
      yaws = limit_period(yaws, 0.5, math.pi)
    else:
      yaws = limit_period(yaws, -1, math.pi)
    return yaws

  @staticmethod
  def check_bboxes(bboxes, yx_zb):
    '''
    x_size > y_size
    '''
    ofs = 1e-6
    if bboxes.shape[0]==0:
      return
    if yx_zb:
      assert torch.all(bboxes[:,3] <= bboxes[:,4])
      assert torch.max(torch.abs(bboxes[:,-1]))<=math.pi*0.5+ofs
    else:
      assert torch.all(bboxes[:,3] >= bboxes[:,4])
      assert torch.max(bboxes[:,-1]) <= math.pi+ofs
      assert torch.min(bboxes[:,-1]) >= 0-ofs
