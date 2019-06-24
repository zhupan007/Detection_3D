import torch
from torch.nn import functional as F
from maskrcnn_benchmark.structures.bounding_box_3d import cat_boxlist_3d

DEBUG = False

class SeperateClassifier():
    def __init__(self, seperate_classes, num_input_classes):
      '''
      (1) For RPN
      Each feature predict two proposals, one for seperated classes, the other one for remaining classes
      (2) For ROI
      Add a background label for the seperated classes at the end. As a result, the dimension of predicted classes increases by 1.
      The dimension of predicted boxes increases by 7.

      0: the seperated classes
      1: the remaining classes
      '''
      self.need_seperate = len(seperate_classes) > 0
      if not self.need_seperate:
        return

      self.num_input_classes = num_input_classes # include background
      self.seperated_num_classes_total = num_input_classes + 1
      seperate_classes.sort()
      assert 0 not in seperate_classes
      self.seperate_classes = seperate_classes = seperate_classes + [self.num_input_classes]
      self.num_classes0 = len(seperate_classes)
      self.remaining_classes = [i for i in range(num_input_classes) if i not in seperate_classes]
      self.num_classes1 = len(self.remaining_classes)
      assert 0 in self.remaining_classes

      self.org_labels_to_labels0 = torch.zeros([num_input_classes+1], dtype=torch.int32)
      self.labels0_to_org_labels = torch.ones([self.num_classes0], dtype=torch.int32) * (-1)
      for i, c in enumerate([0]+seperate_classes[:-1]):
        self.org_labels_to_labels0[c] = i # 0 not in seperate_classes
        self.labels0_to_org_labels[i] = c
      self.labels0_to_org_labels[0] = num_input_classes

      self.org_labels_to_labels1 = torch.zeros([num_input_classes+1], dtype=torch.int32)
      self.labels1_to_org_labels = torch.ones([self.num_classes1], dtype=torch.int32) * (-1)
      # the 0(background) of org_labels is the 0 for remaining classes

      for i, c in enumerate(self.remaining_classes):
        self.org_labels_to_labels1[c] = i
        self.labels1_to_org_labels[i] = c

      if DEBUG:
        print(f'\n\nseperate_classes: {seperate_classes}')
        print(f'labels0_to_org_labels: {self.labels0_to_org_labels}')
        print(f'org_labels_to_labels0: {self.org_labels_to_labels0}')
        print(f'labels1_to_org_labels: {self.labels1_to_org_labels}')
        print(f'org_labels_to_labels1: {self.org_labels_to_labels1}')

      pass

    #---------------------------------------------------------------------------
    # For RPN
    #---------------------------------------------------------------------------
    def seperate_rpn_selector(self, box_selector_fn, anchors, objectness, rpn_box_regression, targets, add_gt_proposals):
      '''
        objectness: [n,2]
        rpn_box_regression: [n,14]
        targets: labels 0~nc_total
        self.targets0: labels 0~nc_0
        self.targets1: labels 0~nc_1
      '''
      self.targets0, self.targets1 = self.seperate_targets_and_update_labels(targets)
      boxes0 = box_selector_fn(anchors, objectness[:,0], rpn_box_regression[:,0:7], self.targets0, add_gt_proposals)
      boxes1 = box_selector_fn(anchors, objectness[:,1], rpn_box_regression[:,7:14], self.targets1, add_gt_proposals)
      boxes = [boxes0, boxes1]

      if DEBUG and False:
        show_box_fields(targets, 'A')
        show_box_fields(self.targets0, 'B')
        show_box_fields(self.targets1, 'C')
        show_box_fields(boxes0, 'D')
        show_box_fields(boxes1, 'E')
        show_box_fields(anchors, 'F')
      return boxes

    def seperate_rpn_loss_evaluator(self, loss_evaluator_fn, anchors, objectness, rpn_box_regression, targets):
      #targets0, targets1 = self.seperate_targets(targets)
      loss_objectness0, loss_rpn_box_reg0 = loss_evaluator_fn(anchors, objectness[:,0], rpn_box_regression[:,0:7], self.targets0)
      loss_objectness1, loss_rpn_box_reg1 = loss_evaluator_fn(anchors, objectness[:,1], rpn_box_regression[:,7:14], self.targets1)
      #loss_objectness = loss_objectness0 + loss_objectness1
      #loss_rpn_box_reg = loss_rpn_box_reg0 + loss_rpn_box_reg1

      loss_objectness = [loss_objectness0, loss_objectness1]
      loss_rpn_box_reg = [loss_rpn_box_reg0, loss_rpn_box_reg1]

      if DEBUG and False:
        show_box_fields(self.targets0, 'B')
        show_box_fields(self.targets1, 'C')
      return loss_objectness, loss_rpn_box_reg

    #---------------------------------------------------------------------------
    # For Detector
    #---------------------------------------------------------------------------
    def sep_roi_heads( self, roi_heads_fn, roi_features, proposals, targets):
      if DEBUG and False:
        show_box_fields(proposals, 'A')
      proposals = self.cat_boxlist_3d_seperated(proposals)
      if DEBUG and False:
        show_box_fields(proposals, 'B')
      return roi_heads_fn(roi_features, proposals, targets)

    #---------------------------------------------------------------------------
    # For ROI
    #---------------------------------------------------------------------------
    def seperate_subsample(self, proposals, targets, subsample_fn):
        proposals_0, proposals_1, _, _ = self.seperate_proposals(proposals)
        self.targets_0, self.targets_1 = self.seperate_targets_and_update_labels(targets)

        proposals_0_ = subsample_fn(proposals_0, self.targets_0)
        proposals_1_ = subsample_fn(proposals_1, self.targets_1)
        bs = len(proposals)
        proposals_out = []
        for i in range(bs):
          proposals_out.append( cat_boxlist_3d([proposals_0_[i], proposals_1_[i]], per_example=False) )

        assert self.targets_0[0].get_field('labels').max() <= self.num_classes0 - 1
        assert self.targets_1[0].get_field('labels').max() <= self.num_classes1 - 1


        if DEBUG and False:
          show_box_fields(proposals, 'In')
          show_box_fields(proposals_0, 'Sep0')
          show_box_fields(proposals_1, 'Sep1')
          show_box_fields(proposals_0_, 'subs0')
          show_box_fields(proposals_1_, 'subs1')
          show_box_fields(proposals_out, 'Out')

          show_box_fields(self.targets_0, 'T0')
          show_box_fields(self.targets_1, 'T1')
        return proposals_out

    def cross_entropy_seperated(self, class_logits, labels, proposals):
      '''
      class_logits: [n, num_classes+1]
      labels: [n]
      self.seperate_classes: [num_classes0] (not include 0)

      In the (num_classes+1) dims of class_logits, the first (num_classes0+1) dims are for self.seperate_classes,
      the following (num_classes1+1) are for the remianing.
      '''
      self.sep_ids0_all_roi, self.sep_ids1_all_roi = self.get_sep_ids_from_proposals(proposals)
      class_logits0, class_logits1 = self.seperate_pred_logits(class_logits, self.sep_ids0_all_roi, self.sep_ids1_all_roi)
      self.labels0_roi = labels[self.sep_ids0_all_roi]
      self.labels1_roi = labels[self.sep_ids1_all_roi]

      assert self.labels0_roi.max() <= self.num_classes0 - 1
      assert self.labels1_roi.max() <= self.num_classes1 - 1

      loss0 = F.cross_entropy(class_logits0, self.labels0_roi)
      loss1 = F.cross_entropy(class_logits1, self.labels1_roi)

      return [loss0, loss1]

    def box_loss_seperated(self, box_loss_fn, labels, box_regression, regression_targets, pro_bbox3ds):
        '''
        labels: [n,2]
        box_regression: [b,7*seperated_num_classes_total]
        regression_targets:[n,7,2]
        pro_bbox3ds:[n,7]
        '''
        box_regression0, box_regression1 = self.seperate_pred_box(box_regression,
                                    self.sep_ids0_all_roi, self.sep_ids1_all_roi)
        regression_targets_0 = regression_targets[self.sep_ids0_all_roi]
        regression_targets_1 = regression_targets[self.sep_ids1_all_roi]
        pro_bbox3ds_0 = pro_bbox3ds[self.sep_ids0_all_roi]
        pro_bbox3ds_1 = pro_bbox3ds[self.sep_ids1_all_roi]
        box_loss0 = box_loss_fn(self.labels0_roi, box_regression0, regression_targets_0, pro_bbox3ds_0)
        box_loss1 = box_loss_fn(self.labels1_roi, box_regression1, regression_targets_1, pro_bbox3ds_1)
        return [box_loss0, box_loss1]
        box_loss = box_loss0 + box_loss1
        return box_loss


    #---------------------------------------------------------------------------
    # Functions Utils
    #---------------------------------------------------------------------------
    def cat_boxlist_3d_seperated(self, bboxes_ls):
        batch_size = bboxes_ls[0].batch_size()
        m = len(bboxes_ls)
        assert m==2

        bboxes_ls[0].add_field('sep_id', torch.zeros([len(bboxes_ls[0])], dtype=torch.int32))
        bboxes_ls[1].add_field('sep_id', torch.ones([len(bboxes_ls[1])], dtype=torch.int32))


        bboxes_ = [None]*m
        for i in range(m):
          bboxes_[i] = bboxes_ls[i].seperate_examples()

        bboxes_ls_new = []
        for j in range(batch_size):
          bboxes_ls_new.append( cat_boxlist_3d([bboxes_[i][j] for i in range(m)], per_example=False) )
        bboxes_ls_new_all = cat_boxlist_3d(bboxes_ls_new, per_example=True)
        return bboxes_ls_new_all

    def seperate_proposals(self, proposals):
      bs = len(proposals)
      proposals_0 = []
      proposals_1 = []
      sep_ids0_all = []
      sep_ids1_all = []
      ids_cum_sum = 0

      for i in range(bs):
        sep_id = proposals[i].get_field('sep_id')
        sep_ids0 = torch.nonzero(1-sep_id).view([-1])
        sep_ids1 = torch.nonzero(sep_id).view(-1)

        sep_ids0_all.append(sep_ids0 + ids_cum_sum)
        sep_ids1_all.append(sep_ids1 + ids_cum_sum)
        ids_cum_sum += len(proposals[i])

        proposals_0.append( proposals[i][sep_ids0] )
        proposals_1.append( proposals[i][sep_ids1] )

      sep_ids0_all = torch.cat(sep_ids0_all, 0)
      sep_ids1_all = torch.cat(sep_ids1_all, 0)
      return proposals_0, proposals_1, sep_ids0_all, sep_ids1_all

    def get_sep_ids_from_proposals(self, proposals):
      bs = len(proposals)
      sep_ids0_all = []
      sep_ids1_all = []
      ids_cum_sum = 0
      for i in range(bs):
        sep_id = proposals[i].get_field('sep_id')
        sep_ids0 = torch.nonzero(1-sep_id).view([-1])
        sep_ids1 = torch.nonzero(sep_id).view(-1)

        sep_ids0_all.append(sep_ids0 + ids_cum_sum)
        sep_ids1_all.append(sep_ids1 + ids_cum_sum)
        ids_cum_sum += len(proposals[i])

      sep_ids0_all = torch.cat(sep_ids0_all, 0)
      sep_ids1_all = torch.cat(sep_ids1_all, 0)
      return sep_ids0_all, sep_ids1_all

    def rm_gt_from_proposals_seperated(self, rm_gt_from_proposals_fn,
              class_logits, box_regression, proposals, targets):

        proposals_0, proposals_1, sep_ids0_all, sep_ids1_all  = self.seperate_proposals(proposals)
        class_logits0, class_logits1 = self.seperate_pred_logits(class_logits, sep_ids0_all, sep_ids1_all)
        box_regression0, box_regression1 = self.seperate_pred_box(box_regression, sep_ids0_all, sep_ids1_all)

        class_logits__0, box_regression__0, proposals__0 =  rm_gt_from_proposals_fn(class_logits0, box_regression0, proposals_0, self.targets_0)
        class_logits__1, box_regression__1, proposals__1 =  rm_gt_from_proposals_fn(class_logits1, box_regression1, proposals_1, self.targets_1)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

    def seperate_pred_logits(self, class_logits, sep_ids0_all, sep_ids1_all):
      assert class_logits.shape[1] == self.seperated_num_classes_total
      assert class_logits.shape[0] == sep_ids0_all.shape[0] + sep_ids1_all.shape[0]
      class_logits0 = class_logits[sep_ids0_all,:] [:,self.seperate_classes]
      class_logits1 = class_logits[sep_ids1_all,:] [:,self.remaining_classes]
      return class_logits0, class_logits1

    def seperate_pred_box(self, box_regression, sep_ids0_all, sep_ids1_all):
      assert box_regression.shape[1] == self.seperated_num_classes_total*7
      assert box_regression.shape[0] == sep_ids0_all.shape[0] + sep_ids1_all.shape[0]
      n = box_regression.shape[0]
      box_regression0 = box_regression.view([n,-1,7])[:, self.seperate_classes, :].view([n,-1])[sep_ids0_all]
      box_regression1 = box_regression.view([n,-1,7])[:, self.remaining_classes, :].view([n,-1])[sep_ids1_all]
      return box_regression0, box_regression1

    def seperate_boxes(self, boxes_ls):
      boxes_ls0 = []
      boxes_ls1 = []
      for boxes in boxes_ls:
          boxes0 = boxes.copy()
          assert set(boxes0.fields()) == set(['objectness', 'labels', 'regression_targets'])
          boxes0.extra_fields['labels'] = boxes0.extra_fields['labels'][:,0]
          boxes0.extra_fields['regression_targets'] = boxes0.extra_fields['regression_targets'][:,:,0]

          boxes1 = boxes.copy()
          boxes1.extra_fields['labels'] = boxes1.extra_fields['labels'][:,1]
          boxes1.extra_fields['regression_targets'] = boxes1.extra_fields['regression_targets'][:,:,1]

          boxes_ls0.append(boxes0)
          boxes_ls1.append(boxes1)
      return boxes_ls0, boxes_ls1

    def _seperating_ids(self, labels):
      assert isinstance(labels, torch.Tensor)
      ids_0s = []
      for c in self.seperate_classes:
        ids_c = torch.nonzero(labels==c).view([-1])
        ids_0s.append(ids_c)
      ids_0 = torch.cat(ids_0s, 0)
      n = labels.shape[0]
      tmp = torch.ones([n])
      tmp[ids_0] = 0
      ids_1 = torch.nonzero(tmp).view([-1])
      return ids_0, ids_1

    def seperate_targets_and_update_labels(self, targets):
      targets0, targets1 = self.seperate_targets(targets)
      bs = len(targets)
      for i in range(bs):
        org_labels0 = targets0[i].get_field('labels')
        labels0 = self.update_labels_to_seperated_id([org_labels0], 0)
        targets0[i].extra_fields['labels'] = labels0[0]
        org_labels1 = targets1[i].get_field('labels')
        labels1 = self.update_labels_to_seperated_id([org_labels1], 1)
        targets1[i].extra_fields['labels'] = labels1[0]
      return targets0, targets1

    def seperate_targets(self, targets):
        assert isinstance(targets, list)
        targets_0 = []
        targets_1 = []
        for tar in targets:
          labels = tar.get_field('labels')
          ids_0, ids_1 = self._seperating_ids(labels)
          tar0 = tar[ids_0]
          tar1 = tar[ids_1]
          targets_0.append( tar0 )
          targets_1.append( tar1 )
        return targets_0, targets_1

    def update_labels_to_seperated_id(self, labels_seperated_org, id):
      '''
      labels_seperated_org: the value is originally value of not seperated, but only part.
      '''
      assert isinstance(labels_seperated_org, list)
      labels_new = []
      device = labels_seperated_org[0].device
      if id==0:
        org_to_new = self.org_labels_to_labels0
      elif id==1:
        org_to_new = self.org_labels_to_labels1
      for ls in labels_seperated_org:
        labels_new.append( org_to_new[ls.long()].to(device).long() )
        if labels_new[-1].shape[0] > 0:
          try:
            assert labels_new[-1].min() >= 0
          except:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
      return labels_new

    def turn_labels_back_to_org(self, result, sep_flag):
      if sep_flag == 0:
        l2ol = self.labels0_to_org_labels
      elif sep_flag == 1:
        l2ol = self.labels1_to_org_labels
      bs = len(result)
      for b in range(bs):
        result[b].extra_fields['labels'] =  l2ol[result[b].extra_fields['labels']]
      return result

    def post_processor(self, class_logits, box_regression, proposals, post_processor_fn):
      proposals_0, proposals_1, sep_ids0_all, sep_ids1_all  = self.seperate_proposals(proposals)
      class_logits0, class_logits1 = self.seperate_pred_logits(class_logits, sep_ids0_all, sep_ids1_all)
      box_regression0, box_regression1 = self.seperate_pred_box(box_regression, sep_ids0_all, sep_ids1_all)

      result0 = post_processor_fn( (class_logits0, box_regression0), proposals_0 )
      result1 = post_processor_fn( (class_logits1, box_regression1), proposals_1 )

      batch_size = len(proposals)
      result = []
      for b in range(batch_size):
        result0[b].extra_fields['labels'] = self.labels0_to_org_labels[result0[b].extra_fields['labels']]
        result1[b].extra_fields['labels'] = self.labels1_to_org_labels[result1[b].extra_fields['labels']]
        result_b = cat_boxlist_3d([result0[b], result1[b]], per_example=False)
        result.append(result_b)

      #print(result[0].fields())
      return result


def show_box_fields(boxes, flag=''):
  print(f'\n\n{flag}')
  if isinstance(boxes, list):
    print(f'bs = {len(boxes)}')
    boxes = boxes[0]
  fields = boxes.fields()
  print(f'size:{len(boxes)} \nfields: {fields}')
  for fie in fields:
    fv = boxes.get_field(fie)
    fv_min = fv.min()
    fv_max = fv.max()
    print(f'{fie}: from {fv_min} to {fv_max}')

