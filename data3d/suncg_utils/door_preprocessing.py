# xyz Arpil 2019


import numpy as np
from utils3d.bbox3d_ops import Bbox3D
from utils3d.geometric_util import limit_period, vertical_dis_1point_lines, angle_of_2lines, vertical_dis_points_lines, ave_angles, lines_intersection_2d
from render_tools import show_walls_offsetz, show_walls_1by1

DEBUG = False

def preprocess_doors(doors0, walls, door_thickness=0.18):
  '''
  doors0:[d,7]
  walls:[w,7]
  '''
  door_n = doors0.shape[0]
  wall_n = walls.shape[0]
  walls_1 = walls.copy()
  # Aug the door to contain door corners
  walls_1[:,4] += 0.2
  # Find the coresponding wall for each door by searching which wall contain two
  # top corners of door.
  door_corners = Bbox3D.bboxes_corners(doors0, 'Z')[:,0:4,:]  # [d,4,3]
  door_lines0 = np.take( door_corners,  Bbox3D._lines_z0_vids, axis=1 ) # [d,4,2,3]
  door_lines1 = door_lines0.reshape(-1,2,3)
  wall_cen_lines = Bbox3D.bboxes_centroid_lines( walls_1,'X','Z' )

  dw_ints = lines_intersection_2d(door_lines1[:,:,0:2], wall_cen_lines[:,:,0:2], True, True)
  dw_ints = dw_ints.reshape(door_n,4,wall_n,2)
  tmp0 = (1-np.isnan(dw_ints)).sum(3)
  tmp1 = tmp0.sum(1)
  assert np.all(tmp1.sum(1)==4), "each door should corespond to 1 wall"
  door_ids, wall_ids = np.where(tmp1)

  walls_good = walls[wall_ids]
  yaws = limit_period(walls_good[:,-1], 0, np.pi/2)

  intersections_2d = []
  for i in range(door_n):
    cids = np.where(tmp0[i,:,wall_ids[i]])[0]
    ints_i  = dw_ints[i, cids, wall_ids[i]]
    intersections_2d.append(ints_i.reshape(1,2,2))
  intersections_2d = np.concatenate(intersections_2d, 0)

  doors_centroids_2d = np.mean(intersections_2d, 1)
  doors_length = np.linalg.norm(intersections_2d[:,0] - intersections_2d[:,1], axis=1)
  doors_length -= door_thickness * np.sin(yaws*2)

  doors_new = doors0.copy()
  doors_new[:,0:2] = doors_centroids_2d
  doors_new[:,3] = doors_length
  doors_new[:,4] = door_thickness
  doors_new[:,6] = walls_good[:,-1]


  if DEBUG:
    show_all([doors0, doors_new, walls])
  return doors_new

  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  dc_in_walls_mask = Bbox3D.points_in_bbox(door_corners.reshape(-1,3), walls_1).reshape(-1,4, walls_1.shape[0]) # [d,4,w]
  door_ids, wall_ids = np.where(dc_in_walls_mask.sum(1) == 2)
  if not np.all(door_ids == np.arange(door_n)):
    print( "Each door has to map one single wall")
    print(f"door_ids:\n{door_ids}")
    fail_ids = np.array([i for i in range(door_n) if i not in door_ids])
    show_all([doors0[fail_ids], walls])
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    assert False

  # the coresponding walls for each door
  walls_dc = walls[wall_ids] # [d,7]

  dc_in_w_mask_1 = []
  for i in range(door_n):
    dc_in_w_mask_1.append( dc_in_walls_mask[i:i+1,:, wall_ids[i]] )
  dc_in_w_mask_1 = np.concatenate(dc_in_w_mask_1, 0)

  # the index of 2 top corners inside walls for each door
  dc_ids = np.where(dc_in_w_mask_1)[1].reshape(-1,2) # [d,2]
  door_corners_good = [] # [d,2,3]
  door_corners_bad = []  # [d,2,3]
  for i in range(door_n):
    door_corners_good.append( door_corners[i:i+1,dc_ids[i]] )
    others_ids = np.array([j for j in range(4) if j not in dc_ids[i]])
    door_corners_bad.append( door_corners[i:i+1, others_ids] )
  door_corners_good = np.concatenate(door_corners_good, 0)
  door_corners_bad = np.concatenate(door_corners_bad, 0)

  new_door_lengthes = np.linalg.norm( door_corners_good[:,1,:] - door_corners_good[:,0,:], axis=1 )

  # thic_v: the unit vector along thickness
  corner_center_good = np.mean(door_corners_good,1)
  corner_center_bad = np.mean(door_corners_bad,1)
  thic_v = corner_center_bad - corner_center_good
  thic_v = thic_v / np.linalg.norm(thic_v, axis=1).reshape(-1,1)

  # The distance from wall centroid to good surface of door
  wall_centroids = walls_dc[:,0:3].copy()
  wall_centroids[:,2] = 0
  dis_wc_gs = []
  for i in range(door_n):
    dis_wc_gs.append( vertical_dis_points_lines(wall_centroids[i:i+1], door_corners_good[i:i+1])[0] )
  dis_wc_gs = np.concatenate(dis_wc_gs, 0)
  door_thickness = dis_wc_gs.reshape(-1,1) * 1.7

  door_centroids = corner_center_good + thic_v*door_thickness*0.5

  new_doors = doors0.copy()
  new_doors[:,0:2] = door_centroids[:,0:2]
  new_doors[:,3] = new_door_lengthes
  new_doors[:,4] = door_thickness[:,0]
  new_doors[:,6] = walls_dc[:,-1]


  if DEBUG:
    show_all([doors0, new_doors, walls_dc])
  return new_doors



def show_all(boxes_ls):
  boxes = np.concatenate(boxes_ls, 0)
  Bbox3D.draw_points_bboxes(boxes[:,0:3], boxes, 'Z', is_yx_zb=False)
