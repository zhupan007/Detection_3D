import torch
from maskrcnn_benchmark.structures.bounding_box_3d import cat_boxlist_3d


class SeperateClassifier():
    def __init__(self, seperate_classes, num_input_classes):
      '''
      Add a background label for the seperated classes at the end
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

      self.org_labels_to_labels0 = torch.ones([num_input_classes+1], dtype=torch.int32) * (-1)
      self.labels0_to_org_labels = torch.ones([self.num_classes0], dtype=torch.int32) * (-1)
      for i, c in enumerate([0]+seperate_classes[:-1]):
        self.org_labels_to_labels0[c] = i # 0 not in seperate_classes
        self.labels0_to_org_labels[i] = c
      self.labels0_to_org_labels[0] = num_input_classes

      self.org_labels_to_labels1 = torch.ones([num_input_classes+1], dtype=torch.int32) * (-1)
      self.labels1_to_org_labels = torch.ones([self.num_classes1], dtype=torch.int32) * (-1)
      # the 0(background) of org_labels is the 0 for remaining classes

      for i, c in enumerate(self.remaining_classes):
        self.org_labels_to_labels1[c] = i
        self.labels1_to_org_labels[i] = c

      pass

    #---------------------------------------------------------------------------
    # For Detector
    #---------------------------------------------------------------------------
    def sep_roi_heads( self, roi_heads_fn0, roi_heads_fn1, roi_features, proposals, targets):
      #assert isinstance(roi_heads_fns, tuple)
      targets0, targets1 = self.seperate_targets_and_update_labels_to_seperated_id(targets)
      #print(targets[0].get_field('labels'))
      #print(targets0[0].get_field('labels'))
      #print(targets1[0].get_field('labels'))
      x0, result0, detector_losses0 = roi_heads_fn0( roi_features, proposals[0], targets0)
      x1, result1, detector_losses1 = roi_heads_fn1( roi_features, proposals[1], targets1)

      #print(result0[0].fields())
      #print(result0[0].get_field('labels'))
      #print(result1[0].get_field('labels'))

      for key in detector_losses0:
        detector_losses0[key] += detector_losses1[key]
      detector_losses = detector_losses0

      result0 = self.turn_labels_back_to_org(result0, 0)
      result1 = self.turn_labels_back_to_org(result1, 1)

      #print(result0[0].get_field('labels'))
      #print(result1[0].get_field('labels'))

      bs = len(targets)
      result = []
      for i in range(bs):
        result.append( cat_boxlist_3d([result0[i], result1[i]], per_example=False) )
      return x0, result, detector_losses

    #---------------------------------------------------------------------------
    # For RPN
    #---------------------------------------------------------------------------
    def seperate_selector(self, box_selector_fn, anchors, objectness, rpn_box_regression, targets, add_gt_proposals):
      targets0, targets1 = self.seperate_targets(targets)
      boxes0 = box_selector_fn(anchors, objectness[:,0], rpn_box_regression[:,0:7], targets0, add_gt_proposals)
      boxes1 = box_selector_fn(anchors, objectness[:,1], rpn_box_regression[:,7:14], targets1, add_gt_proposals)
      #boxes = cat_boxlist_3d([boxes0, boxes1], per_example=False)
      boxes = (boxes0, boxes1)
      return boxes

    def loss_evaluator(self, loss_evaluator_fn, anchors, objectness, rpn_box_regression, targets):
      targets0, targets1 = self.seperate_targets(targets)
      loss_objectness0, loss_rpn_box_reg0 = loss_evaluator_fn(anchors, objectness[:,0], rpn_box_regression[:,0:7], targets0)
      loss_objectness1, loss_rpn_box_reg1 = loss_evaluator_fn(anchors, objectness[:,1], rpn_box_regression[:,7:14], targets0)
      loss_objectness = loss_objectness0 + loss_objectness1
      loss_rpn_box_reg = loss_rpn_box_reg0 + loss_rpn_box_reg1
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      return loss_objectness, loss_rpn_box_reg


    #---------------------------------------------------------------------------
    # For ROI
    #---------------------------------------------------------------------------

    def seperate_pred_logits(self, class_logits):
      assert class_logits.shape[1] == self.seperated_num_classes_total
      class_logits0 = class_logits[:, self.seperate_classes]
      class_logits1 = class_logits[:, self.remaining_classes]
      return class_logits0, class_logits1

    def seperate_pred_box(self, box_regression):
      assert box_regression.shape[1] == self.seperated_num_classes_total*7
      n = box_regression.shape[0]
      box_regression0 = box_regression.view([n,-1,7])[:, self.seperate_classes, :].view([n,-1])
      box_regression1 = box_regression.view([n,-1,7])[:, self.remaining_classes, :].view([n,-1])
      return box_regression0, box_regression1

    def seperate_boxes(self, boxes_ls):
      boxes_ls0 = []
      boxes_ls1 = []
      for boxes in boxes_ls:
          boxes0 = boxes.copy()
          assert boxes0.fields() == ['objectness', 'labels', 'regression_targets']
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

    def seperate_targets_and_update_labels_to_seperated_id(self, targets):
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
        assert labels_new[-1].min() >= 0
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
      class_logits0, class_logits1 = self.seperate_pred_logits(class_logits)
      box_regression0, box_regression1 = self.seperate_pred_box(box_regression)
      proposals0, proposals1 = self.seperate_boxes(proposals)

      result0 = post_processor_fn( (class_logits0, box_regression0), proposals0 )
      result1 = post_processor_fn( (class_logits1, box_regression1), proposals1 )

      batch_size = len(proposals)
      result = []
      for b in range(batch_size):
        result0[b].extra_fields['labels'] = self.labels0_to_org_labels[result0[b].extra_fields['labels']]
        result1[b].extra_fields['labels'] = self.labels1_to_org_labels[result1[b].extra_fields['labels']]
        result_b = cat_boxlist_3d([result0[b], result1[b]], per_example=False)
        result.append(result_b)

      #print(result[0].fields())
      return result

