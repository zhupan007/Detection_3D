import sys, os, glob, json
sys.path.insert(0, '..')
import open3d, pymesh
import numpy as np
from utils3d.bbox3d_ops import Bbox3D
from utils3d.geometric_util import cam2world_box, cam2world_pcl
import torch
from collections import defaultdict

SUNCG_V1_DIR = '/DS/SUNCG/suncg_v1'
PARSED_DIR = f'{SUNCG_V1_DIR}/parsed'
SPLITED_DIR = '/DS/SUNCG/suncg_v1_splited_torch'

CLASSES = ['wall', 'window', 'door']
#CLASSES = ['door']

def show_walls_1by1(wall_bboxes):
  n = wall_bboxes.shape[0]
  for i in range(n):
    tmp = wall_bboxes.copy()
    tmp[:,2] -= 1
    show_box = np.concatenate([tmp, wall_bboxes[i:i+1]], 0)
    print(f'wall {i}/{n}')
    Bbox3D.draw_bboxes(show_box, 'Z', False)

def show_walls_offsetz(wall_bboxes):
  n = wall_bboxes.shape[0]
  wall_bboxes = wall_bboxes.copy()
  wall_bboxes[:,2] += np.random.rand(n)*2
  print(f'totally {n} walls')
  Bbox3D.draw_bboxes(wall_bboxes, 'Z', False)


def cut_points_roof(points, keep_rate=0.95):
  z_min = np.min(points[:,2])
  z_max = np.max(points[:,2])
  threshold = z_min + (z_max - z_min) * keep_rate
  mask = points[:,2] < threshold
  points_cutted = points[mask]
  return points_cutted

def down_sample_points(points, keep_rate=0.3):
  n = points.shape[0]
  choices = np.random.choice(n, int(n*keep_rate), replace=False)
  points_d = points[choices]
  return points_d

def render_parsed_house_walls(parsed_dir, show_pcl=False):
  bboxes = []
  for obj in CLASSES:
    bbox_fn_ = f'{parsed_dir}/object_bbox/{obj}.txt'
    bboxes_  = np.loadtxt(bbox_fn_).reshape([-1,7])
    bboxes.append(bboxes_)
  bboxes = np.concatenate(bboxes, 0)

  #Bbox3D.draw_bboxes(bboxes, up_axis='Z', is_yx_zb=False)
  Bbox3D.draw_bboxes_mesh(bboxes, up_axis='Z', is_yx_zb=False)

  if show_pcl:
    pcl_fn = f'{parsed_dir}/pcl_camref.ply'
    pcd = open3d.read_point_cloud(pcl_fn)
    points = np.asarray(pcd.points)
    points = cam2world_pcl(points)
    #points = down_sample_points(points, 0.03)
    points = cut_points_roof(points)

    bboxes[:,2] += 0.1
    points = cut_points_roof(points)
    Bbox3D.draw_points_bboxes(points, bboxes, up_axis='Z', is_yx_zb=False)
    Bbox3D.draw_points_bboxes_mesh(points, bboxes, up_axis='Z', is_yx_zb=False)

def render_splited_house_walls(pth_fn):
  pcl, bboxes = torch.load(pth_fn)
  points = pcl[:,0:3]
  points = cut_points_roof(points)

  classes = [k for k in bboxes.keys()]
  num_classes = {k:bboxes[k].shape[0] for k in bboxes.keys()}
  print(f'\nclasses: {num_classes}\n\n')

  all_bboxes = np.concatenate([boxes for boxes in bboxes.values()], 0)
  Bbox3D.draw_points_bboxes(points, all_bboxes, up_axis='Z', is_yx_zb=False)

  #for clas in bboxes.keys():
  #  if clas not in CLASSES:
  #    continue
  #  boxes = bboxes[clas]
  #  Bbox3D.draw_points_bboxes(points, boxes, up_axis='Z', is_yx_zb=False)
  #  #Bbox3D.draw_points_bboxes_mesh(points, boxes, up_axis='Z', is_yx_zb=False)

def render_suncg_raw_house_walls(house_fn):
    from suncg import split_room_parts, Suncg
    with open(house_fn) as f:
      house = json.loads(f.read())

    scaleToMeters = house['scaleToMeters']
    assert scaleToMeters == 1
    bboxes = defaultdict(list)
    bboxes['house'].append( Bbox3D.bbox_from_minmax( house['bbox'] ))

    for level in house['levels']:
      if 'bbox' not in level:
        continue
      bboxes['level'].append( Bbox3D.bbox_from_minmax( level['bbox'] ))
      nodes = level['nodes']
      for node in nodes:
        node_type = node['type']
        if node_type == 'Object':
          modelId = node['modelId']
          category = Suncg.modelId_2_class[modelId]
          bboxes[category].append(Bbox3D.bbox_from_minmax( node['bbox']))
        elif node_type == 'Room':
          if 'bbox' in node:
            bboxes['room'].append(Bbox3D.bbox_from_minmax( node['bbox']))
          room_bboxes = split_room_parts(house_fn, node['modelId'])
          for c in room_bboxes:
            bboxes[c] += room_bboxes[c]
        else:
          if 'bbox' in node:
            bboxes[node_type].append(Bbox3D.bbox_from_minmax( node['bbox']))

    centroid = (np.array(house['bbox']['min']) + np.array(house['bbox']['max']))/2.0
    mesh_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = centroid)

    for obj in bboxes:
      bboxes[obj] = np.concatenate([b.reshape([1,7]) for b in bboxes[obj]], 0)
      bboxes[obj] = cam2world_box(bboxes[obj])
    walls = bboxes['wall']

    print('\nThe raw SUNCG walls\n')
    #Bbox3D.draw_bboxes(walls, up_axis='Z', is_yx_zb=False)
    Bbox3D.draw_bboxes_mesh(walls, up_axis='Z', is_yx_zb=False)

def render_cam_positions(parsed_dir):
  from suncg import save_cam_ply
  cam_fn = f'{parsed_dir}/cam'
  save_cam_ply(cam_fn, show=True, with_pcl=True)


def render_houses(r_cam=True, r_whole=True, r_splited=True):
  #house_names = os.listdir(PARSED_DIR)
  house_names = ['8c033357d15373f4079b1cecef0e065a']
  #house_names = ['28297783bce682aac7fb35a1f35f68fa']
  for house_name in house_names:
    print(f'{house_name}')
    raw_house_fn = f'{SUNCG_V1_DIR}/house/{house_name}/house.json'
    #render_suncg_raw_house_walls(raw_house_fn)

    parsed_dir = f'{PARSED_DIR}/{house_name}'

    if r_cam:
      render_cam_positions(parsed_dir)

    if r_whole:
      render_parsed_house_walls(parsed_dir, True)

    splited_boxfn = f'{SPLITED_DIR}/houses/{house_name}/*.pth'
    pth_fns = glob.glob(splited_boxfn)
    if r_splited:
      for pth_fn in pth_fns:
        print('The splited scene')
        render_splited_house_walls(pth_fn)


if __name__ == '__main__':
  render_houses(r_cam=False, r_whole=False, r_splited=True)

