import sys, os, glob, json
sys.path.insert(0, '..')
import open3d, pymesh
import numpy as np
from utils3d.bbox3d_ops import Bbox3D
from utils3d.geometric_util import cam2world_box, cam2world_pcl
import torch
from collections import defaultdict
from suncg_utils.scene_samples import SceneSamples
from data3d.dataset_metas import DSET_METAS0

SUNCG_V1_DIR = '/DS/SUNCG/suncg_v1'
PARSED_DIR = f'{SUNCG_V1_DIR}/parsed'
SPLITED_DIR = '/DS/SUNCG/suncg_v1_torch_splited'

#CLASSES = ['wall', 'ceiling']
CLASSES = ['wall', 'window', 'door']
#CLASSES = ['ceiling', 'floor']
CLASSES += ['floor']
#CLASSES += ['room']

SHOW_PCL = 1
POINTS_KEEP_RATE = 0.3
NOT_SHOW_CEILING = True

def show_walls_1by1(wall_bboxes):
  n = wall_bboxes.shape[0]
  for i in range(n):
    tmp = wall_bboxes.copy()
    tmp[:,2] -= 1
    show_box = np.concatenate([tmp, wall_bboxes[i:i+1]], 0)
    print(f'wall {i}/{n}\n{wall_bboxes[i]}')
    Bbox3D.draw_bboxes(show_box, 'Z', False)

def show_walls_offsetz(wall_bboxes):
  n = wall_bboxes.shape[0]
  wall_bboxes = wall_bboxes.copy()
  wall_bboxes[:,2] += np.random.rand(n)*1
  print(f'totally {n} boxes')
  Bbox3D.draw_bboxes(wall_bboxes, 'Z', False)


def cut_points_roof(points, keep_rate=0.9):
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


def render_parsed_house_walls(parsed_dir, show_pcl=SHOW_PCL, show_by_class=1):
  print(f'parsed_dir:{parsed_dir}')
  bboxes = []
  labels = []
  for obj in CLASSES:
    bbox_fn_ = f'{parsed_dir}/object_bbox/{obj}.txt'
    bboxes_  = np.loadtxt(bbox_fn_).reshape([-1,7])
    bboxes.append(bboxes_)
    label = DSET_METAS0.class_2_label[obj]
    labels += [label] * bboxes_.shape[0]
  bboxes = np.concatenate(bboxes, 0)
  labels = np.array(labels).astype(np.int8)
  if bboxes.shape[0] > 0:
    scene_size = Bbox3D.boxes_size(bboxes)
    print(f'scene wall size:{scene_size}')

    #Bbox3D.draw_bboxes(bboxes, up_axis='Z', is_yx_zb=False, labels=labels)
    #if not show_pcl:
    #Bbox3D.draw_bboxes_mesh(bboxes, up_axis='Z', is_yx_zb=False)
    Bbox3D.draw_bboxes_mesh(bboxes, up_axis='Z', is_yx_zb=False, labels=labels)
    #show_walls_offsetz(bboxes)

    if show_by_class:
          for c in range(1,max(labels)+1):
              cs = DSET_METAS0.label_2_class[c]
              print(cs)
              if cs not in ['wall', 'window', 'door']:
              #if cs not in ['wall']:
                continue
              mask = labels == c
              bboxes_c = bboxes[mask]
              show_walls_offsetz(bboxes_c)

  if show_pcl:
    pcl_fn = f'{parsed_dir}/pcl_camref.ply'
    if not os.path.exists(pcl_fn):
        return

    pcd = open3d.read_point_cloud(pcl_fn)
    points = np.asarray(pcd.points)
    points = cam2world_pcl(points)
    colors = np.asarray(pcd.colors)
    pcl = np.concatenate([points, colors], 1)

    scene_size = pcl_size(pcl)
    print(f'scene pcl size:{scene_size}')
    print(f'point num: {pcl.shape[0]}')

    #pcl = cut_points_roof(pcl)

    bboxes[:,2] += 0.1
    Bbox3D.draw_points_bboxes(pcl, bboxes, up_axis='Z', is_yx_zb=False)
    #Bbox3D.draw_points_bboxes_mesh(pcl, bboxes, up_axis='Z', is_yx_zb=False)



def pcl_size(pcl):
    xyz_max = pcl[:,0:3].max(0)
    xyz_min = pcl[:,0:3].min(0)
    xyz_size = xyz_max - xyz_min
    return xyz_size

def render_pth_file(pth_fn, show_by_class=False):
  pcl, bboxes = torch.load(pth_fn)
  #points = pcl[:,0:3]
  #colors = pcl[:,3:6]
  #normals = pcl[:,6:9]

  if NOT_SHOW_CEILING:
    if 'ceiling' in bboxes:
      del bboxes['ceiling']
    if 'room' in bboxes:
      del bboxes['room']

  scene_size = pcl_size(pcl)
  print(f'scene pcl size:{scene_size}')
  print(f'point num: {pcl.shape[0]}')

  #pcl = cut_points_roof(pcl)

  classes = [k for k in bboxes.keys()]
  num_classes = {k:bboxes[k].shape[0] for k in bboxes.keys()}
  print(f'\nclasses: {num_classes}\n\n')

  all_bboxes = np.concatenate([boxes for boxes in bboxes.values()], 0)
  nums = [boxes.shape[0] for boxes in bboxes.values()]
  labels = []
  for i, n in enumerate(nums):
    labels += [i]*n
  labels = np.array(labels)


  #show_walls_offsetz(all_bboxes)
  Bbox3D.draw_bboxes_mesh(all_bboxes, up_axis='Z', is_yx_zb=False, labels=labels)
  Bbox3D.draw_points_bboxes(pcl, all_bboxes, up_axis='Z', is_yx_zb=False, labels=labels, points_keep_rate=POINTS_KEEP_RATE)

  if show_by_class:
    for clas in bboxes.keys():
      print(clas)
      #if clas not in CLASSES:
      #  continue
      boxes = bboxes[clas]
      #Bbox3D.draw_points_bboxes(points, boxes, up_axis='Z', is_yx_zb=False)
      Bbox3D.draw_points_bboxes_mesh(pcl, boxes, up_axis='Z', is_yx_zb=False, points_keep_rate=POINTS_KEEP_RATE)

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
  '''
  angle%90 != 0:
        72148738e98fe68f38ec17945d5c9730 *
        b021ab18bb170a167d569dcfcaf58cd4 *
        8c033357d15373f4079b1cecef0e065a **
        b021ab18bb170a167d569dcfcaf58cd4 ** small angle
  complicate architecture:
      31a69e882e51c7c5dfdc0da464c3c02d **
  '''
  house_names = ['b021ab18bb170a167d569dcfcaf58cd4'] #
  house_names = ['0163e180b8c372c9a7f123dc01ae43ed']

  #house_names = os.listdir(PARSED_DIR)

  with open(f'{SUNCG_V1_DIR}/house_names_1level.txt', 'r') as h1f:
      house_names_1level = h1f.read().split('\n')
  house_names = house_names_1level

  house_names.sort()
  print(f'totally {len(house_names)} houses')

  #house_names = house_names[320:]

  #house_names = SceneSamples.very_hard_wall_window_close
  #house_names = ['03ca5b7fcc114833b18cd5f22ae0edf8']

  for k,house_name in enumerate( house_names ):
    print(f'\n{k}: {house_name}')
    raw_house_fn = f'{SUNCG_V1_DIR}/house/{house_name}/house.json'
    #render_suncg_raw_house_walls(raw_house_fn)

    parsed_dir = f'{PARSED_DIR}/{house_name}'

    if r_cam:
      render_cam_positions(parsed_dir)

    if r_whole:
      render_parsed_house_walls(parsed_dir)

    splited_boxfn = f'{SPLITED_DIR}/houses/{house_name}/*.pth'
    pth_fns = glob.glob(splited_boxfn)
    if r_splited:
      for i,pth_fn in enumerate( pth_fns ):
        print(f'\nThe {i}-th / {len(pth_fns)} splited scene')
        render_pth_file(pth_fn)


def render_fn():
    pth_fn = '/DS/SUNCG/suncg_v1_splited_torch_BS_30_30_BN_300K/houses/0005b50577f5871e1c0bb7a687f6cbc3/pcl_0.pth'
    render_pth_file(pth_fn)

def main():
    render_houses(
            r_cam=False,
            r_whole = 0,
            r_splited = 1
    )



if __name__ == '__main__':
    #render_fn()
    main()



