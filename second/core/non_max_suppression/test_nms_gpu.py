from nms_gpu  import rotate_iou_gpu_eval
import numpy as np

def main():
  boxes = np.array([
    [0, 0, 1,   2., 0.1],
    [0, 0, .001,   2., 0.1],
    [0, 0, 0.1, 2., 0.5],
    [0, 0, 0.1, 2., -np.pi/2],
    ])

  boxes = np.array([
    [ 7.2792,  0.2153,  0.0947,  1.4407,   -1.5708],
    [ 6.5114,  0.1272,  0.0947,  0.2544,    0.0000]
    ])

  boxes = np.array([
        [ 2.3569,  7.0700, -0.0300,  0.0947,  1.8593,  2.7350,  0.0000],
        [ 1.1548,  6.1797, -0.0300,  0.0947,  2.3096,  2.7350, -1.5708]
    ])

  ious = np.diag( rotate_iou_gpu_eval(boxes, boxes) )
  print(f"ious: {ious}")
  #old: [0.         0.         0.33333316 0.        ]
  #new: [1.         0.99998605 0.99999934 1.        ]

if __name__ == '__main__':
  main()
