# xyz  April 2019

import numpy as np
from utils3d.bbox3d_ops import Bbox3D
from utils3d.geometric_util import limit_period, vertical_dis_1point_lines, angle_of_2lines, vertical_dis_points_lines, ave_angles
from render_tools import show_walls_offsetz, show_walls_1by1

DEBUG = False

def preprocess_windows(windows0, walls):
  '''
  both windows0 ad walls are standard: [xc, yc, zc, x_size, y_size, z_size, yaw]
  '''
  #Bbox3D.draw_bboxes(walls, 'Z', False)
  #Bbox3D.draw_bboxes(windows0, 'Z', False)
  #print(f'windows0: \n{windows0}')
  if len(windows0) == 0:
    return windows0
  windows1 = Bbox3D.define_walls_direction(windows0, 'Z', yx_zb=False)
  #print(f'windows1: \n{windows0}')
  win_bad_ids, wall_ids_for_bad_win  = find_wall_ids_for_windows(windows1, walls)

  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  windows_bad = windows1[win_bad_ids].reshape(-1,7)
  walls_bw = walls[wall_ids_for_bad_win].reshape(-1,7)
  windows_corrected = correct_bad_windows(windows_bad, walls_bw)
  windows1[win_bad_ids] = windows_corrected

  if DEBUG:
    show_all([windows1,walls])
  return windows1

def find_wall_ids_for_windows(windows, walls):
  win_in_walls = Bbox3D.points_in_bbox(windows[:,0:3], walls)
  wall_ids = np.where(win_in_walls)[1]
  wall_yaws = walls[wall_ids, -1]
  yaw_bad = (np.abs(wall_yaws) / (np.pi*0.5)) % 1 > 0.01
  win_bad_ids = np.where(yaw_bad)[0]
  wall_ids_for_bad_win = wall_ids[win_bad_ids]
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  return win_bad_ids, wall_ids_for_bad_win


def correct_bad_windows(windows_bad, walls):
  windows_cor = windows_bad.copy()
  windows_cor[:,-1] = walls[:,-1]
  windows_cor[:,4] = 0.175
  windows_cor[:,3] = np.sqrt(windows_bad[:,3]**2 + windows_bad[:,4]**2)
  yaws = limit_period(walls[:,-1], 0, np.pi/2)
  windows_cor[:,3] -= 0.175 * np.sin(yaws*2)
  if DEBUG and True:
    show_all([windows_bad, windows_cor, walls])
  return windows_cor

def show_all(boxes_ls):
  boxes = np.concatenate(boxes_ls, 0)
  Bbox3D.draw_points_bboxes(boxes[:,0:3], boxes, 'Z', is_yx_zb=False)
