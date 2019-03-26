# Copyright (c) Facebook, BoxList3DInc. and its affiliates. All Rights Reserved.
import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def cat_boxlist_3d(bboxes, per_example=False):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
        per_example: if True, each element in bboxes is an example, combine to a batch
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList3D) for bbox in bboxes)

    if not per_example:
      size3d = bboxes[0].size3d
      for bbox3d in bboxes:
        #is_size_close =  torch.abs(bbox3d.size3d - size3d).max() < 0.01
        #if not is_size_close:
        if not torch.isclose( bbox3d.size3d, size3d ).all():
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
    else:
      size3d = torch.cat([b.size3d for b in bboxes])

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    batch_size0 = bboxes[0].batch_size()
    for bbox in bboxes:
      assert bbox.batch_size() == batch_size0

    bbox3d_cat = _cat([bbox3d.bbox3d for bbox3d in bboxes], dim=0)
    if not per_example:
      examples_idxscope = torch.tensor([[0, bbox3d_cat.shape[0]]], dtype=torch.int32)
      batch_size = batch_size0
      assert batch_size0 == 1, "check if >1 if need to"
    else:
      assert batch_size0 == 1, "check if >1 if need to"
      batch_size = len(bboxes)
      examples_idxscope = torch.cat([b.examples_idxscope for b in bboxes])
      for b in range(1,batch_size):
        examples_idxscope[b,:] += examples_idxscope[b-1,1]
    cat_boxes = BoxList3D(bbox3d_cat, size3d, mode, examples_idxscope)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def cat_scales_anchor(anchors):
    '''
     combine anchors of scales
     anchors: list(BoxList)

     anchors_new: BoxList
     final flatten order: [batch_size, scale_num, yaws_num, sparse_feature_num]
    '''
    scale_num = len(anchors)
    batch_size = anchors[0].batch_size()
    anchors_scales = []
    for s in range(scale_num):
      anchors_scales.append( anchors[s].seperate_examples() )

    num_examples = [[len(an) for an in ans] for ans in anchors_scales] # [batch_size, scale_num]


    examples = []
    for b in range(batch_size):
      examples.append( cat_boxlist_3d([a[b] for a in anchors_scales] ) )
    anchors_all_scales = cat_boxlist_3d(examples, per_example=True)
    return anchors_all_scales


class BoxList3D(object):
    """
    This class represents a set of 3d bounding boxes.
    The bounding boxes are represented as a Nx7 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox3d, size3d, mode, examples_idxscope):
        '''
        All examples in same batch are concatenated together.
        examples_idxscope: [batch_size,2] record the index scope per example
        bbox3d: [N,7] N=sum(bbox num of each example)
        size3d: None or [N,6] -> [6] [is xyz_min, xyz_max]
        mode: "standard", "yx_zb"
        extra_fields: "label" "objectness"
        '''
        assert bbox3d.shape[1] == 7, bbox3d.shape
        assert examples_idxscope[-1,-1] == bbox3d.shape[0]
        if size3d is not None:
          assert size3d.shape[1] == 6
          assert  size3d.shape[0] == examples_idxscope.shape[0]

        device = bbox3d.device if isinstance(bbox3d, torch.Tensor) else torch.device("cpu")
        bbox3d = torch.as_tensor(bbox3d, dtype=torch.float32, device=device)
        if bbox3d.ndimension() != 2:
            raise ValueError(
                "bbox3d should have 2 dimensions, got {}".format(bbox3d.ndimension())
            )
        if bbox3d.size(-1) != 7:
            raise ValueError(
                "last dimenion of bbox3d should have a "
                "size of 7, got {}".format(bbox3d.size(-1))
            )
        self.check_mode(mode)

        self.bbox3d = bbox3d
        self.size3d = size3d
        self.mode = mode
        self.examples_idxscope = examples_idxscope
        self.extra_fields = {}

    def check_mode(self, mode):
        if mode not in ("standard", "yx_zb"):
            raise ValueError("mode should be 'standard' or 'yx_zb'")

    def batch_size(self):
        return self.examples_idxscope.shape[0]
    def seperate_examples(self):
      batch_size = self.batch_size()
      examples = []
      for bi in range(batch_size):
        examples.append(self.example(bi))
      return examples

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        self.check_mode(mode)
        if mode == self.mode:
            return self
        bbox3d0 = self.bbox3d
        bbox3d1 = bbox3d0[:,[0,1,2,4,3,5,6]]
        if mode == 'standard':
          bbox3d1[:,2] += bbox3d0[:,5] * 0.5
        else:
          bbox3d1[:,2] -= bbox3d0[:,5] * 0.5
        bbox = BoxList3D(bbox3d1, self.size3d, mode, self.examples_idxscope)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox3d = BoxList3D(self.bbox3d.to(device), self.size3d.to(device), self.mode, self.examples_idxscope)

        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox3d.add_field(k, v)
        return bbox3d

    def example(self, idx):
        assert idx < self.batch_size()
        se = self.examples_idxscope[idx]
        examples_idxscope = torch.tensor([[0, se[1]-se[0]]], dtype=torch.int32)
        bbox3d = BoxList3D( self.bbox3d[se[0]:se[1],:], self.size3d[idx:idx+1], self.mode, examples_idxscope)
        for k, v in self.extra_fields.items():
            bbox3d.add_field(k, v[item][se[0]:se[1]])
        return bbox3d

    def __getitem__(self, item):
        assert self.batch_size() == 1
        if isinstance(item, torch.Tensor):
          assert len(item.shape) == 1
        else:
          if isinstance(item, int):
            item = [item]
          assert isinstance(item, list)
          item = torch.tensor(item, dtype=torch.int64)
        examples_idxscope = torch.tensor([[0,item.shape[0]]], dtype=torch.int32)
        bbox3d = BoxList3D(self.bbox3d[item], self.size3d, self.mode, examples_idxscope)
        for k, v in self.extra_fields.items():
            bbox3d.add_field(k, v[item])
        return bbox3d

    def __len__(self):
        return self.bbox3d.shape[0]

    def clip_to_pcl(self, remove_empty=True):
        return
        raise NotImplementedError
        TO_REMOVE = 1
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        self.bbox3d[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox3d[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox3d[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox3d[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields):
        bbox3d = BoxList3D(self.bbox3d, self.size3d, self.mode, self.examples_idxscope)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            bbox3d.add_field(field, self.get_field(field))
        return bbox3d

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "mode={})".format(self.mode)
        return s


    def show(self, max_num=-1, points=None, with_centroids=False):
      import numpy as np
      from utils3d.bbox3d_ops import Bbox3D
      boxes = self.bbox3d.cpu().data.numpy()
      if with_centroids:
        centroids = boxes.copy()
        centroids[:,3:6] = 0.02
      if max_num > 0 and max_num < boxes.shape[0]:
        ids = np.random.choice(boxes.shape[0], max_num, replace=False)
        boxes = boxes[ids]
      if with_centroids:
        boxes = np.concatenate([boxes, centroids], 0)
      if points is None:
        Bbox3D.draw_bboxes(boxes, 'Z', is_yx_zb=self.mode=='yx_zb')
      else:
        Bbox3D.draw_points_bboxes(points, boxes, 'Z', is_yx_zb=self.mode=='yx_zb', random_color=False)

    def show_centroids(self, max_num=-1, points=None):
      import numpy as np
      from utils3d.bbox3d_ops import Bbox3D
      boxes = self.bbox3d.cpu().data.numpy()
      if max_num > 0 and max_num < boxes.shape[0]:
        ids = np.random.choice(boxes.shape[0], max_num, replace=False)
        boxes = boxes[ids]
      if points is None:
        Bbox3D.draw_centroids(boxes, 'Z', is_yx_zb=self.mode=='yx_zb')
      else:
        Bbox3D.draw_points_centroids(points, boxes, 'Z', is_yx_zb=self.mode=='yx_zb')

    def show_together(self, boxlist_1, max_num=-1, max_num_1=-1, points=None):
      import numpy as np
      from utils3d.bbox3d_ops import Bbox3D
      boxes = self.bbox3d.cpu().data.numpy().copy()
      if max_num > 0 and max_num < boxes.shape[0]:
        ids = np.random.choice(boxes.shape[0], max_num, replace=False)
        boxes = boxes[ids]

      boxes_1 = boxlist_1.bbox3d.cpu().data.numpy()
      if max_num_1 > 0 and max_num_1 < boxes_1.shape[0]:
        ids = np.random.choice(boxes_1.shape[0], max_num_1, replace=False)
        boxes_1 = boxes_1[ids]

      labels = np.array([0]*boxes.shape[0] + [1]*boxes_1.shape[0])
      boxes = np.concatenate([boxes, boxes_1], 0)

      if points is None:
        Bbox3D.draw_bboxes(boxes, 'Z', is_yx_zb=self.mode=='yx_zb', labels=labels)
      else:
        if isinstance(points, torch.Tensor):
          points = points.cpu().data.numpy()
        Bbox3D.draw_points_bboxes(points, boxes, 'Z', is_yx_zb=self.mode=='yx_zb', labels=labels, random_color=False)

if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
