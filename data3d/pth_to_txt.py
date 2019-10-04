import torch, os
import numpy as np

def pth_to_txt(pth_fn):
  '''
    class_2_label0 = {'background':0,'wall':1, 'window':2, 'door':3,
                    'ceiling':5, 'floor': 4, 'room':6}
  '''
  res = torch.load(pth_fn)
  bboxes = [r.bbox3d for r in res]
  labels = [r.get_field('labels') for r in res]
  n = len(bboxes)
  path = os.path.dirname(pth_fn) + '/text_models'
  if not os.path.exists(path):
    os.makedirs(path)
  for i in range(n):
    txt_fn = os.path.join(path,f'room_{i}.txt')
    room = np.concatenate([bboxes[i], labels[i].reshape(-1,1)], 1)
    np.savetxt(txt_fn, room)
    print(f'save {txt_fn} ok')
  pass

if __name__ == '__main__':
  pth_fn = '/home/z/Research/Detection_3D/demo/suncg_test_5_iou_3_augth_2/predictions.pth'
  pth_to_txt(pth_fn)
