# May 2018 xyz
import numpy as np
import numba

def Rx( x ):
    # ref to my master notes 2015
    # anticlockwise, x: radian
    Rx = np.zeros((3,3))
    Rx[0,0] = 1
    Rx[1,1] = np.cos(x)
    Rx[1,2] = np.sin(x)
    Rx[2,1] = -np.sin(x)
    Rx[2,2] = np.cos(x)
    return Rx

def Ry( y ):
    # anticlockwise, y: radian
    Ry = np.zeros((3,3))
    Ry[0,0] = np.cos(y)
    Ry[0,2] = -np.sin(y)
    Ry[1,1] = 1
    Ry[2,0] = np.sin(y)
    Ry[2,2] = np.cos(y)
    return Ry

@numba.jit(nopython=True)
def Rz( z ):
    # anticlockwise, z: radian
    Rz = np.zeros((3,3))

    Rz[0,0] = np.cos(z)
    Rz[0,1] = np.sin(z)
    Rz[1,0] = -np.sin(z)
    Rz[1,1] = np.cos(z)
    Rz[2,2] = 1
    return Rz

def R1D( angle, axis ):
    if axis == 'x':
        return Rx(angle)
    elif axis == 'y':
        return Ry(angle)
    elif axis == 'z':
        return Rz(angle)
    else:
        raise NotImplementedError

def EulerRotate( angles, order ='zxy' ):
    R = np.eye(3)
    for i in range(3):
        R_i = R1D(angles[i], order[i])
        R = np.matmul( R_i, R )
    return R

def point_rotation_randomly( points, rxyz_max=np.pi*np.array([0.1,0.1,0.1]) ):
    # Input:
    #   points: (B, N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (B, N, 3)
    batch_size = points.shape[0]
    for b in range(batch_size):
        rxyz = [ np.random.uniform(-r_max, r_max) for r_max in rxyz_max ]
        R = EulerRotate( rxyz, 'xyz' )
        points[b,:,:] = np.matmul( points[b,:,:], np.transpose(R) )
    return points

def angle_with_x(direc, scope_id=0):
  if direc.ndim == 2:
    x = np.array([[1.0,0]])
  if direc.ndim == 3:
    x = np.array([[1.0,0,0]])
  x = np.tile(x, [direc.shape[0],1])
  return angle_of_2lines(direc, x, scope_id)

def angle_of_2lines(line0, line1, scope_id=0):
  '''
    line0: [n,2/3]
    line1: [n,2/3]
    zero as ref

   scope_id=0: [0,pi]
            1: (-pi/2, pi/2]

   angle: [n]
  '''
  assert line0.ndim == line1.ndim == 2
  assert (line0.shape[0] == line1.shape[0]) or line0.shape[0]==1 or line1.shape[0]==1
  assert line0.shape[1] == line1.shape[1] # 2 or 3

  norm0 = np.linalg.norm(line0, axis=1, keepdims=True)
  norm1 = np.linalg.norm(line1, axis=1, keepdims=True)
  line0 = line0 / norm0
  line1 = line1 / norm1
  angle = np.arccos( np.sum(line0 * line1, axis=1) )

  if scope_id == 0:
    pass
  elif scope_id == 1:
    # (-pi/2, pi/2]: offset=0.5, period=pi
    angle = limit_period(angle, 0.5, np.pi)
  else:
    raise NotImplementedError
  return angle

def limit_period(val, offset, period):
  '''
   [0, pi]: offset=0, period=pi
    [-pi/2, pi/2]: offset=0.5, period=pi
    [-pi, 0]: offset=1, period=pi
  '''
  return val - np.floor(val / period + offset) * period

def ave_angles(angles, scope_id=0):
  '''
    angles: [n,2]
    scope_id = 0: [-pi/2, pi/2]
    period = np.pi

    make the angle between the average and both are below half period

    out: [n]
  '''
  assert angles.ndim == 2
  assert angles.shape[1] == 2

  period = np.pi
  dif = angles[:,1] - angles[:,0]
  mask = np.abs(dif) > period * 0.5
  angles[:,1] += - period * mask * np.sign(dif)
  ave = angles.mean(axis=1)
  if scope_id==0:
    ave = limit_period(ave, 0.5, period)
  else:
    raise NotImplementedError
  return ave

def vertical_dis_points_lines(points, lines):
  '''
  points:[n,3]
  lines:[m,2,3]
  dis: [n,m]
  '''
  dis = []
  pn = points.shape[0]
  for i in range(pn):
    dis.append( vertical_dis_1point_lines(points[i], lines).reshape([1,-1]) )
  dis = np.concatenate(dis, 0)
  return dis

def vertical_dis_1point_lines(point, lines):
  '''
  point:[3]
  lines:[m,2,3]
  dis: [m]
  '''
  assert point.ndim == 1
  assert lines.ndim == 3
  assert lines.shape[1:] == (2,3) or lines.shape[1:] == (2,2)
  # use lines[:,0,:] as ref
  point = point.reshape([1,-1])
  ln = lines.shape[0]
  direc_p = point - lines[:,0,:]
  direc_l = lines[:,1,:] - lines[:,0,:]
  angles = angle_of_2lines(direc_p, direc_l, scope_id=0)
  dis = np.sin(angles) * np.linalg.norm(direc_p, axis=1)
  return dis

def cam2world_pcl(points):
  R = np.eye(points.shape[-1])
  R[1,1] = R[2,2] = 0
  R[1,2] = 1
  R[2,1] = -1
  points = np.matmul(points, R)
  return points

def cam2world_box(box):
  assert box.shape[1] == 7
  R = np.eye(7)
  R[1,1] = R[2,2] = 0
  R[1,2] = 1
  R[2,1] = -1
  R[4,4] = R[5,5] = 0
  R[4,5] = 1
  R[5,4] = 1
  R[6,6] = 1
  box = np.matmul(box, R)
  return box


class OBJ_DEF():
  @staticmethod
  def limit_yaw(yaws, yx_zb):
    '''
    standard: [0, pi]
    yx_zb: [-pi/2, pi/2]
    '''
    if yx_zb:
      yaws = limit_period(yaws, 0.5, np.pi)
    else:
      yaws = limit_period(yaws, 0, np.pi)
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
      #assert np.all(bboxes[:,3] <= bboxes[:,4]) # prediction may not mathch
      assert np.max(np.abs(bboxes[:,-1])) <= np.pi*0.5+ofs
    else:
      #assert np.all(bboxes[:,3] >= bboxes[:,4])
      assert np.max(bboxes[:,-1]) <= np.pi + ofs
      assert np.min(bboxes[:,-1]) >= 0 - ofs
