import sys, os, glob, json
sys.path.insert(0, '..')
import open3d, pymesh
import numpy as np
from utils3d.bbox3d_ops import Bbox3D
from utils3d.geometric_util import cam2world_box, cam2world_pcl
import torch
from collections import defaultdict
from suncg_utils.suncg_meta import SUNCG_META

SUNCG_V1_DIR = '/DS/SUNCG/suncg_v1'
PARSED_DIR = f'{SUNCG_V1_DIR}/parsed'
SPLITED_DIR = '/DS/SUNCG/suncg_v1_torch_splited'

#CLASSES = ['wall', 'ceiling']
CLASSES = ['wall', 'window', 'door']
#CLASSES = ['ceiling', 'floor']
CLASSES += ['floor']
#CLASSES += ['room']

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
  wall_bboxes[:,2] += np.random.rand(n)*1
  print(f'totally {n} walls')
  Bbox3D.draw_bboxes(wall_bboxes, 'Z', False)


def cut_points_roof(points, keep_rate=0.55):
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
  labels = []
  for obj in CLASSES:
    bbox_fn_ = f'{parsed_dir}/object_bbox/{obj}.txt'
    bboxes_  = np.loadtxt(bbox_fn_).reshape([-1,7])
    bboxes.append(bboxes_)
    label = SUNCG_META.class_2_label[obj]
    labels += [label] * bboxes_.shape[0]
  bboxes = np.concatenate(bboxes, 0)
  labels = np.array(labels).astype(np.int8)
  scene_size = Bbox3D.boxes_size(bboxes)
  print(f'scene wall size:{scene_size}')

  #Bbox3D.draw_bboxes(bboxes, up_axis='Z', is_yx_zb=False, labels=labels)
  #if not show_pcl:
  Bbox3D.draw_bboxes_mesh(bboxes, up_axis='Z', is_yx_zb=False)
  #Bbox3D.draw_bboxes_mesh(bboxes, up_axis='Z', is_yx_zb=False, labels=labels)
  show_walls_offsetz(bboxes)

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

    pcl = cut_points_roof(pcl)

    bboxes[:,2] += 0.1
    Bbox3D.draw_points_bboxes(pcl, bboxes, up_axis='Z', is_yx_zb=False)
    #Bbox3D.draw_points_bboxes_mesh(pcl, bboxes, up_axis='Z', is_yx_zb=False)

def pcl_size(pcl):
    xyz_max = pcl[:,0:3].max(0)
    xyz_min = pcl[:,0:3].min(0)
    xyz_size = xyz_max - xyz_min
    return xyz_size

def render_pth_file(pth_fn):
  pcl, bboxes = torch.load(pth_fn)
  #points = pcl[:,0:3]
  #colors = pcl[:,3:6]
  #normals = pcl[:,6:9]

  scene_size = pcl_size(pcl)
  print(f'scene pcl size:{scene_size}')
  print(f'point num: {pcl.shape[0]}')

  pcl = cut_points_roof(pcl)

  classes = [k for k in bboxes.keys()]
  num_classes = {k:bboxes[k].shape[0] for k in bboxes.keys()}
  print(f'\nclasses: {num_classes}\n\n')

  all_bboxes = np.concatenate([boxes for boxes in bboxes.values()], 0)
  Bbox3D.draw_points_bboxes(pcl, all_bboxes, up_axis='Z', is_yx_zb=False)
  show_walls_offsetz(all_bboxes)

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
  house_names = ['31a69e882e51c7c5dfdc0da464c3c02d']
  #house_names = ['72148738e98fe68f38ec17945d5c9730']
  #house_names = ['8c033357d15373f4079b1cecef0e065a']
  #house_names = ['7411df25770eaf8d656cac2be42a9af0']
  #house_names = ['7cd75b127f06a078929a6524396c738c']
  #house_names = ['e7b3e2566e174b6fbb2864de76b50334']
  #house_names = ['aaa535ef80b7d34f57f5d3274eec0daf']
  house_names = ['c3802ae080bc1d5f4ada2f75448f7b49']
  house_names = ['be37c58e21c4595b3cf3ccaf3cbc51c4']
  house_names = ['3a86005157c0acf437626cde8e26b4be']
  house_names = ['a046e442fa9c38ae063e8ea9d2ceeeea']
  house_names = ['8c033357d15373f4079b1cecef0e065a']

  house_names = os.listdir(PARSED_DIR)
  house_names.sort()
  for k,house_name in enumerate( house_names ):
    print(f'\n{k}: {house_name}')
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
      for i,pth_fn in enumerate( pth_fns ):
        print(f'\nThe {i}-th / {len(pth_fns)} splited scene')
        render_pth_file(pth_fn)


def render_obj_house():
    import pymesh
    folder = '/DS/SUNCG/suncg_v1/parsed/31a69e882e51c7c5dfdc0da464c3c02d'
    folder = '/home/z/SUNCG/suncg_v1/parsed/31a69e882e51c7c5dfdc0da464c3c02d'
    fn = f'{folder}/house.obj'
    mesh = pymesh.load_mesh(fn)
    new_mesh_fn = f'{folder}/new_house.obj'

    vertices = mesh.vertices
    faces = mesh.faces
    attributes = {}

    mask = np.arange(10000)
    faces = faces[mask]
    for an in mesh.get_attribute_names():
        attributes[an] = mesh.get_face_attribute(an)[mask]

    new_mesh = pymesh.form_mesh(vertices, faces)
    for an in attributes:
        new_mesh.add_attribute(an)
        new_mesh.set_attribute(an, attributes[an])
    pymesh.save_mesh(new_mesh_fn, new_mesh, use_float=True)


def render_fn():
    pth_fn = '/home/z/Research/Detection_3D/data3d/suncg_utils/SuncgTorch/houses/a046e442fa9c38ae063e8ea9d2ceeeea/pcl_1.pth'
    pth_fn = '/DS/SUNCG/suncg_v1_splited_torch_BS_30_30_BN_300K/houses/8c033357d15373f4079b1cecef0e065a/pcl_0.pth'
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



