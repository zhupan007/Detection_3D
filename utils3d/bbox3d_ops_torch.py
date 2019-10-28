# xyz Oct 2019
import torch, math
from utils3d.geometric_torch import angle_with_x
from utils3d.geometric_torch import OBJ_DEF

_cx,_cy,_cz, _sx,_sy,_sz, _yaw = range(7)
class Box3D_Torch():
  '''
      bbox yx_zb  : [xc, yc, z_bot, y_size, x_size, z_size, yaw-0.5pi]
      bbox standard: [xc, yc, zc, x_size, y_size, z_size, yaw]
      bbox corner : [x0, y0, x1, y1, z0, z1, thickness]
      y_size is thickness
  '''
  @staticmethod
  def corner_box_to_yxzb(boxes):
    assert boxes.shape[1] == 7
    boxes_stand = Box3D_Torch.corner_box_to_standard(boxes)
    boxes_yxzb = Box3D_Torch.convert_from_yx_zb_boxes(boxes_stand)
    return boxes_yxzb

  @staticmethod
  def corner_box_to_standard(boxes):
    centroid = (boxes[:,[0,1,4]] + boxes[:,[2,3,5]])/2.0
    box_direction = boxes[:,[2,3]] - boxes[:,[0,1]]
    xsize = torch.norm( box_direction, dim=1 )
    xsize = xsize.view([-1,1])
    ysize = boxes[:,6:7]
    zsize = (boxes[:,5] - boxes[:,4]).view([-1, 1])
    yaw = angle_with_x( box_direction, scope_id=1 ).view([-1,1])
    boxes_stand = torch.cat([centroid, xsize, ysize, zsize, yaw], 1)
    return boxes_stand

  @staticmethod
  def convert_from_yx_zb_boxes(boxes):
    '''
    Input
      bbox yx_zb  : [xc, yc, z_bot, y_size, x_size, z_size, yaw-0.5pi]
    Output
      bbox standard: [xc, yc, zc, x_size, y_size, z_size, yaw]

    The input is kitti lidar bbox format used in SECOND: x,y,z,w,l,h,orientation
      orientation=0: positive x of camera/car = negative lidar y -> car front face neg lidar y
      orientation = -pi/2: car face pos x of world -> clock wise rotation is positive
      orientation : (-pi,0]


    In my standard definition, bbox frame is same as world -> yaw=0. Also clock wise is positive.
    yaw = pi/2 is the orientation=0 status for yx_zb format of SECOND.
    yaw: (-pi/2,pi/2]

    yaw = orientation + pi/2

    The output format is the standard format I used in Bbox3D
    '''
    assert boxes.shape[1] == 7
    boxes = boxes.clone().detach()
    if boxes.shape[0] == 0:
      return boxes
    boxes[:,2] += boxes[:,5]*0.5
    boxes = boxes[:,[0,1,2,4,3,5,6]]
    boxes[:,-1] += math.pi*0.5
    # limit in [-pi/2, pi/2]
    boxes[:,_yaw] = OBJ_DEF.limit_yaw(boxes[:,_yaw], False)
    OBJ_DEF.check_bboxes(boxes, False)
    return boxes

  @staticmethod
  def convert_to_yx_zb_boxes(boxes):
    '''
    Input
      bbox standard
    Output
      bbox yx_zb
    '''
    assert boxes.shape[1] == 7

    # This should be implemented in data prepration. For ceiling, floor, room,
    # temporaly performed here.
    #boxes = Bbox3D.define_walls_direction(boxes, 'Z', yx_zb=False, check_thickness=False)

    boxes = boxes[:,[0,1,2,4,3,5,6]]
    boxes[:,2] = boxes[:,2] - boxes[:,5]*0.5
    boxes[:,-1] -= math.pi*0.5
    boxes[:,_yaw] = OBJ_DEF.limit_yaw(boxes[:,_yaw], True)
    OBJ_DEF.check_bboxes(boxes, True)
    return boxes

