from utils3d.bbox3d_ops import Bbox3D
import numpy as np
from utils3d.geometric_util import vertical_dis_points_lines, points_in_lines, is_extend_lines
from data3d.render_tools import show_walls_offsetz, show_walls_1by1

Debug = 0

def preprocess_cfr(ceilings_org, walls_org, obj):
  '''
    Z is up, Y is thickness
    A ceiling is good:
      (1) not contains multiple rooms:
        cover no other ceilings
      (2) >= 3 edge walls
    A edge wall of a ceiling:
      In three points of the wall cenline: two corners and centroid
      at least two are on an edge of ceiling
  '''
  assert ceilings_org.ndim == walls_org.ndim == 2
  dis_threshold = 0.07
  if ceilings_org.shape[0] == 0:
    return ceilings_org
  if walls_org.shape[0] == 0:
    return np.zeros(shape=[0,7], dtype=np.float32)

  #Bbox3D.draw_bboxes(walls_org, 'Z', False)
  #Bbox3D.draw_bboxes(ceilings_org, 'Z', False)
  ceilings = ceilings_org.copy()
  ceilings, keep_ids0 = clean_repeat(ceilings)
  walls = walls_org.copy()
  cn = ceilings.shape[0]

  ceilings[:,2] = 0
  walls[:,2] = 0
  walls[:,5] = 0

  ceil_corners0 = Bbox3D.bboxes_corners(ceilings, 'Z')
  ceil_corners = np.take(ceil_corners0, Bbox3D._zpos_vs, axis=1)
  ceil_corners[:,:,2] = 0

  ceiling_cens = ceilings[:,0:3]
  wall_cenlines = Bbox3D.bboxes_centroid_lines(walls, 'X', 'Z')

  good_ceiling_ids0 = []
  bad_small_ids = []

  for c in range(cn):
    # (1)
    #print(f'c:{c}')
    ceil_c =  ceilings[c:c+1].copy()
    ceil_c[:,3:6] += 0.2
    mask_overlap = Bbox3D.points_in_bbox(ceil_corners.reshape([-1,3]), ceil_c).reshape([cn, 4])
    mask_overlap = mask_overlap.all(1)
    num_overlap = mask_overlap.sum() - 1
    ol_ids = np.where(mask_overlap)[0]
    ol_ids = [i for i in ol_ids if i != c]
    #if any_overlap:
    #  area_ol = np.product(tmp[ol_ids,3:5], axis=1).sum()
    #  area_c = np.product(ceilings[c,3:5])
    #  area_ol_rate = area_ol / area_c
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #  pass
    if Debug  and 0:
          print(f'ceiling, contain {num_overlap} other celings')
          box_show = np.concatenate([walls_org, ceilings_org[c:c+1]], 0)
          Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)
    if num_overlap > 1:
      continue
    #if num_overlap >0:
    #  bad_small_ids += ol_ids

    # (2)
    edge_wall_num, winc_state = is_edge_wall_of_ceiling(wall_cenlines, ceilings[c], walls_org)
    if edge_wall_num >= 3 or (edge_wall_num==2 and (winc_state==3).all()):
      good_ceiling_ids0.append(c)

  #good_ceiling_ids1 = [i for i in good_ceiling_ids0 if i not in bad_small_ids]
  good_ceiling_ids1 = keep_ids0[ good_ceiling_ids0 ]
  good_ceiling_ids1 = np.array(good_ceiling_ids1).astype(np.int)
  rm_num = cn - good_ceiling_ids1.shape[0]

  bad_ceiling_ids = np.array([i for i in range(cn) if i not in good_ceiling_ids1   ]).astype(np.int32)
  new_ceilings = ceilings_org[good_ceiling_ids1]

  if Debug and rm_num>0:
        print(f'{cn} -> {good_ceiling_ids1.shape[0]}')
        box_show = np.concatenate([walls_org, ceilings_org[good_ceiling_ids1]], 0)
        Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)
        if rm_num>0:
          box_show = np.concatenate([walls_org, ceilings_org[bad_ceiling_ids]], 0)
          Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        #show_walls_1by1(new_ceilings)
        pass
  return new_ceilings


def clean_extend_lines(lines):
    '''
    [n,2,3]
    '''
    mask = is_extend_lines(lines, lines)
    n = lines.shape[0]
    rm_ids = []
    for i in range(n-1):
      if mask[i,i+1:].any():
        rm_ids.append(i)
    keep_ids = [i for i in range(n) if i not in rm_ids]
    return keep_ids

def is_edge_wall_of_ceiling(wall_cenlines, ceiling, walls_org):

    c_corners = Bbox3D.bbox_corners(ceiling, 'Z')
    c_lines = np.take(c_corners, Bbox3D._lines_z0_vids, axis=0)
    c_lines[:,:,2] = 0
    wall_cenlines_ = np.concatenate([wall_cenlines, wall_cenlines.mean(1, keepdims=True)], 1)
    wn = wall_cenlines.shape[0]
    in_mask0 = points_in_lines(wall_cenlines_.reshape([-1,3]), c_lines, threshold_dis=0.1) # [wn*3,4]
    in_mask0 = in_mask0.reshape([wn,3,4]) # [wn,3,4]

    # find all wdge walls
    in_mask1 = in_mask0.any(2) # [wn,3] in any of four edge of a ceiling
    in_state = in_mask1.sum(1)
    winc_mask = in_state>=2 # [wn] the number of points on edge: 0~3.  two of three: two corners, one centroid
    winc_ids = np.where(winc_mask)[0]

    # clean repeat edge walls on same edge: only at most one edge wall on each
    # edge allowed
    if winc_ids.shape[0] == 0:
      winc_ids_1 = winc_ids
      winc_state = []
    else:
      keep_ids = clean_extend_lines(wall_cenlines[winc_ids])
      winc_ids_1 = winc_ids[keep_ids]
      winc_state = in_state[winc_ids_1]

    edge_wall_num = winc_ids_1.shape[0]


    if Debug and edge_wall_num<2 and 0:
    #if Debug:
      print(f'edge_wall_num: {edge_wall_num}')
      boxes = np.concatenate([ceiling.reshape([1,-1]), walls_org[winc_ids]], 0)
      wo = walls_org.copy()
      wo[:,2] -= 2
      #boxes = np.concatenate([boxes, wo],0)
      #boxes = ceiling.reshape([1,-1])
      points = np.concatenate([wall_cenlines_.reshape([-1,3]), c_lines.reshape([-1,3])], 0)
      Bbox3D.draw_points_bboxes(points, boxes, 'Z', False)

      boxes = np.concatenate([ceiling.reshape([1,-1]), walls_org[winc_ids_1]], 0)
      Bbox3D.draw_points_bboxes(points, boxes, 'Z', False)
      #show_walls_1by1(walls_org)
    return edge_wall_num, winc_state

def preprocess_cfr_old(ceilings_org, walls_org, obj):
  '''
  Z is up, Y is thickness
  '''
  #Bbox3D.draw_bboxes(walls, 'Z', False)
  #Bbox3D.draw_bboxes(ceilings, 'Z', False)
  ceilings = ceilings_org.copy()
  walls = walls_org.copy()
  dis_threshold = 0.07

  ceiling_cens = ceilings[:,0:3]
  ceiling_cens[:,2] = 0
  ceil_cenlines_x = Bbox3D.bboxes_centroid_lines(ceilings, 'X', 'Z')
  ceil_cenlines_x[:,:,2] = 0
  #ceil_cenlines_y = Bbox3D.bboxes_centroid_lines(ceilings, 'Y', 'Z')
  wall_cenlines = Bbox3D.bboxes_centroid_lines(walls, 'X', 'Z')
  wall_cenlines[:,:,2] = 0


  ceilings_shrink = ceilings.copy()
  ceilings_shrink[:,3:5] -= 0.3

  cn = ceilings.shape[0]

  ## Find edge wall nums
  good_ceiling_ids = []
  for c in range(cn):
    # (0.1) If no any other overlap ceiling, try to keep it
    # Otherwise, delete it when  >3 wall inside ceiling
    tmp = np.delete( ceiling_cens.copy(), c, axis=0 )
    any_overlap = Bbox3D.points_in_bbox(tmp, ceilings[c:c+1]).any()
    if any_overlap:
      wall_corner0_in_ceil = Bbox3D.points_in_bbox(wall_cenlines[:,0,:], ceilings_shrink[c:c+1])
      wall_corner1_in_ceil =  Bbox3D.points_in_bbox(wall_cenlines[:,1,:], ceilings_shrink[c:c+1])
      wall_inside_ceil = wall_corner0_in_ceil + wall_corner1_in_ceil
      wall_inside_ceil_ids = np.where(wall_inside_ceil)[0]
      nwic = wall_inside_ceil_ids.shape[0]

      if nwic > 3:
        if Debug and 1:
          print(f'bad ceiling, contain {nwic} walls inside')
          box_show = np.concatenate([walls_org, ceilings_org[c:c+1]], 0)
          Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)

          box_show = np.concatenate([walls_org[wall_inside_ceil_ids], ceilings_org[c:c+1]], 0)
          Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        continue

    # (1) the central corners of wall are inside of ceiling
    wall_cenlines_auged = line_aug(wall_cenlines)
    cw_cen_dis = ceiling_cens[c].reshape([1,1,-1]) - wall_cenlines_auged
    cw_cen_dis = np.linalg.norm(cw_cen_dis, axis=2)
    ceil_diag_size  = np.linalg.norm( ceilings[c,3:5] )
    on_inside_ceil = (cw_cen_dis - ceil_diag_size/2 < dis_threshold).sum(1) >= 2

    if Debug and 0:
      #Bbox3D.draw_points_bboxes(wall_cenlines_auged.reshape([-1,3]), walls, 'Z', False)
      inside_ids = np.where(on_inside_ceil)[0]
      box_show = np.concatenate([walls_org[inside_ids], ceilings_org[c:c+1]], 0)
      Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)

    # (2) along x: wall central line is on x boundaries of ceiling
    dis_cw = vertical_dis_points_lines(ceil_cenlines_x[c], wall_cenlines)
    ceil_y_thickness = ceilings[c,4]
    mask_x0 = np.abs(dis_cw[0] - dis_cw[1]) < dis_threshold
    mask_x1 = (np.abs(dis_cw - ceil_y_thickness/2) < dis_threshold).all(0)
    mask_x = mask_x0 * mask_x1 * on_inside_ceil
    wall_on_ceil_boundary_parall_x = np.where( mask_x )[0]
    num_edgew_x = clean_edge_wall_same_side(wall_cenlines[wall_on_ceil_boundary_parall_x])

    # (3) along x: wall central line is on x boundaries of ceiling
    ceil_x_thickness = ceilings[c,3]
    mask_y0 = dis_cw  < dis_threshold
    mask_y1 = np.abs(dis_cw - ceil_x_thickness) < dis_threshold
    mask_y = (mask_y0 + mask_y1).all(0)
    mask_y = mask_y * on_inside_ceil
    wall_on_ceil_boundary_parall_y = np.where(mask_y )[0]
    num_edgew_y = clean_edge_wall_same_side(wall_cenlines[wall_on_ceil_boundary_parall_y])

    #Bbox3D.point_in_box(wall_cenlines, ceilings[])

    edge_wall_num = num_edgew_x + num_edgew_y

    if edge_wall_num >= 3:
      good_ceiling_ids.append(c)


    if Debug and edge_wall_num < 3 and 0:
      print(f'edge_wall_num: {edge_wall_num}')
      box_show = np.concatenate([walls_org, ceilings_org[c:c+1]], 0)
      Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)
      #Bbox3D.draw_points_bboxes(ceil_cenlines_x[c], box_show, 'Z', False)
      #Bbox3D.draw_points_bboxes(ceil_cenlines_x[c], ceilings[c:c+1], 'Z', False)

      edge_walls_x = walls_org[wall_on_ceil_boundary_parall_x]
      box_x = np.concatenate([edge_walls_x, ceilings_org[c:c+1]], 0)
      #Bbox3D.draw_bboxes_mesh(box_x, 'Z', False)

      edge_walls_y = walls_org[wall_on_ceil_boundary_parall_y]
      box_y = np.concatenate([edge_walls_y, ceilings_org[c:c+1]], 0)
      #Bbox3D.draw_bboxes_mesh(box_y, 'Z', False)

      walls_inside = walls_org[wall_inside_ceil_ids]
      box_ins = np.concatenate([walls_inside, ceilings_org[c:c+1]], 0)
      #Bbox3D.draw_bboxes_mesh(box_ins, 'Z', False)

      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  good_ceiling_ids = np.array(good_ceiling_ids).reshape([-1])
  new_cn = good_ceiling_ids.shape[0]
  print(f'\n\n{obj} {cn} -> {new_cn}')
  if new_cn == 0:
    new_ceilings = ceilings_org[0:0]
  else:
    new_ceilings = ceilings_org[good_ceiling_ids]
  if Debug and new_cn < cn:
      print(good_ceiling_ids)
      box_show = np.concatenate([walls_org, new_ceilings], 0)
      Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)

      bad_ceil_ids = np.array([i for i in range(cn) if i not in good_ceiling_ids   ]).astype(np.int32)
      if bad_ceil_ids.shape[0]>0:
        box_show = np.concatenate([walls_org, ceilings_org[bad_ceil_ids]], 0)
        Bbox3D.draw_bboxes_mesh(box_show, 'Z', False)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
  return ceilings

def line_aug(wall_cenlines):
  cen = wall_cenlines.mean(axis=1).reshape(-1,1,3)
  aug_points = wall_cenlines * 0.6 + cen * 0.4
  wall_cenlines_auged = np.concatenate([wall_cenlines, aug_points], 1)
  return wall_cenlines_auged

def clean_edge_wall_same_side(cenlines):
  '''
  [n,2,3]
  '''
  n = cenlines.shape[0]
  if n<=1:
    return n
  rm_ids = []
  for i in range(n-1):
    centroid_i = cenlines[i].mean(0).reshape([1,3])
    dis = vertical_dis_points_lines(centroid_i, cenlines[i+1:])
    if dis.min() < 0.05:
      rm_ids.append(i)
  return  n - len(rm_ids)

def clean_repeat(ceilings):
  #Bbox3D.draw_ceilings(ceilings, 'Z', False)
  n = ceilings.shape[0]
  keep_ids = [0]
  for i in range(1,n):
    keep_i = True
    for j in range(i):
      dif = ceilings[i] - ceilings[j]
      cen_dis = np.linalg.norm(dif[0:3])
      ref = max(ceilings[i,3:6].max(),ceilings[j,3:6].max())
      size_dif = np.abs(dif[3:6]).max() / ref
      angle_dif = np.abs(dif[-1])
      is_same_ij = cen_dis < 0.1 and size_dif < 0.1 and angle_dif < 0.1
      if  is_same_ij:
        keep_i = False
        break
    if keep_i:
      keep_ids.append(i)

  keep_ids = np.array(keep_ids, dtype=np.int)
  ceilings = ceilings[keep_ids]
  return ceilings, keep_ids


