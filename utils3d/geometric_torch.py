## April 2019 xyz
import torch, math

def limit_period(val, offset, period):
  '''
   [0, pi]: offset=0, period=pi
    [-pi/2, pi/2]: offset=0.5, period=pi
    [-pi, 0]: offset=1, period=pi
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

def angle_with_x(vec0, scope_id=0):
  vec_x = vec0.clone().detach()
  vec_x[:,0] = 1
  vec_x[:,1:] = 0
  return angle_from_vec0_to_vec1(vec_x, vec0, scope_id)


def angle_from_vec0_to_vec1(vec0, vec1, scope_id=0):
  '''
    vec0: [n,2/3]
    vec1: [n,2/3]
    zero as ref

   scope_id=0: [0,pi]
            1: (-pi/2, pi/2]

   angle: [n]
  '''
  assert vec0.dim() == vec1.dim() == 2
  assert (vec0.shape[0] == vec1.shape[0]) or vec0.shape[0]==1 or vec1.shape[0]==1
  assert vec0.shape[1] == vec1.shape[1] # 2 or 3

  norm0 = torch.norm(vec0, dim=1, keepdim=True)
  norm1 = torch.norm(vec1, dim=1, keepdim=True)
  #assert norm0.min() > 1e-4 and norm1.min() > 1e-4 # return nan
  vec0 = vec0 / norm0
  vec1 = vec1 / norm1
  if vec0.dim() == 2:
    tmp = vec0[:,0:1]*0
    vec0 = torch.cat([vec0, tmp], 1)
    vec1 = torch.cat([vec1, tmp], 1)
  cz = torch.cross( vec0, vec1, dim=1)[:,2]
  angle = torch.asin(cz)
  # cross is positive for anti-clock wise. change to clock-wise
  angle = -angle

  if scope_id == 1:
    pass
  elif scope_id == 0:
    # (0, pi]: offset=0.5, period=pi
    angle = limit_period(angle, 0, np.pi)
  else:
    raise NotImplementedError
  return angle

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
      yaws = limit_period(yaws, 0, math.pi)
    return yaws

  @staticmethod
  def check_bboxes(bboxes, yx_zb, check_thickness=0):
    '''
    x_size > y_size
    '''
    ofs = 1e-6
    if bboxes.shape[0]==0:
      return
    if yx_zb:
      #if not torch.all(bboxes[:,3] <= bboxes[:,4]):
      #  xydif = bboxes[:,3] - bboxes[:,4]
      #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
      #  pass
      #assert torch.all(bboxes[:,3] <= bboxes[:,4])


      #print(bboxes)
      #if not torch.max(torch.abs(bboxes[:,-1]))<=math.pi*0.5+ofs:
      #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
      #  pass
      assert torch.max(torch.abs(bboxes[:,-1]))<=math.pi*0.5+ofs
    else:
      if check_thickness:
        assert torch.all(bboxes[:,3] >= bboxes[:,4])
      if not torch.max(bboxes[:,-1]) <= math.pi+ofs:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
      assert torch.max(bboxes[:,-1]) <= math.pi+ofs
      assert torch.min(bboxes[:,-1]) >= 0-ofs

