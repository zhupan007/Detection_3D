import  glob, os
path = './res_Sw4c_3S_fpn432_bs1_T3'
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
