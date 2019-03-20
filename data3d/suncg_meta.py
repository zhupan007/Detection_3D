
class SUNCG_META():
  class_2_label = {'wall':0, 'window':1, 'door':2}
  label_2_class = {c:o for c,o in zip(class_2_label.values(), class_2_label.keys())}
