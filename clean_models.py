import  glob, os
path = './res543_2d_bs1_wall_rpn'
f = open(f'{path}/last_checkpoint', 'r')
checkpoint = './'+f.readlines()[0]
fnames = glob.glob(f'{path}/model_*.pth')
final = f'{path}/model_final.pth'
for s in fnames:
  if s == checkpoint or s==final:
    continue
  os.remove(s)
  print(f'{s} removed')
print('clean ok')
