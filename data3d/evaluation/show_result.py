import os, torch
from data3d.evaluation.suncg.suncg_eval import show_pred, draw_recall_precision_score

RES_PATH = '/home/z/Research/Detection_3D/RES/res_sw4c_fpn432_bs1_lr5_T6711/inference_3d/suncg_test_2_iou_2'
RES_PATH = '/home/z/Research/Detection_3D/RES/res_sw4c_fpn432_bs1_lr5_T6655/inference_3d/suncg_test_1605_iou_2'
RES_PATH = '/home/z/Research/Detection_3D/RES/res_sw4c_fpn432_bs1_lr5_T6655/inference_3d/suncg_test_1605_iou_5'
RES_PATH = '/home/z/Research/Detection_3D/RES/res_sw4c_fpn432_bs1_lr5_T6655/inference_3d/suncg_test_1605_iou_3_augth_0.2'

def show_prediction():
  pred_fn = os.path.join(RES_PATH, 'preds.pth')
  gt_boxlists_, pred_boxlists_, files = torch.load(pred_fn)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  show_pred(gt_boxlists_, pred_boxlists_, files)

def show_performance():
  pred_fn = os.path.join(RES_PATH, 'performance_res.pth')
  result = torch.load(pred_fn)
  draw_recall_precision_score(result, RES_PATH)


if __name__ == '__main__':
  #show_prediction()
  show_performance()

