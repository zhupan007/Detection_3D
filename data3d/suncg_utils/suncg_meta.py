
class SUNCG_META():
  class_2_label = {'background':0,'wall':1, 'window':2, 'door':3}
  label_2_class = {c:o for c,o in zip(class_2_label.values(), class_2_label.keys())}
