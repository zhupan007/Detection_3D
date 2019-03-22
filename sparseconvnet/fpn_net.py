# xyz Jan 2019

import torch
import torch.nn as nn
import sparseconvnet as scn
from .sparseConvNetTensor import SparseConvNetTensor
import numpy as np

DEBUG = False


class FPN_Net(torch.nn.Module):
    _show = DEBUG
    def __init__(self, full_scale, dimension, raw_elements, reps, nPlanesF, nPlaneM, residual_blocks,
                  fpn_scales, downsample=[[2,2,2], [2,2,2]], leakiness=0):
        '''
        downsample:[kernel, stride]
        '''
        nn.Module.__init__(self)

        self.down_kernels =  downsample[0]
        self.down_strides = downsample[1]
        self.fpn_scales = fpn_scales
        scale_num = len(nPlanesF)
        assert len(self.down_kernels) == scale_num - 1 == len(self.down_strides), f"nPlanesF len = {scale_num}, kernels num = {len(self.down_kernels)}"
        assert all([len(ks)==3 for ks in self.down_kernels])
        assert all([len(ss)==3 for ss in self.down_strides])
        self._merge = 'add'  # 'cat' or 'add'

        ele_channels = {'xyz':3, 'normal':3, 'color':3}
        in_channels = sum([ele_channels[e] for e in raw_elements])

        self.layers_in = scn.Sequential(
                scn.InputLayer(dimension,full_scale, mode=4),
                scn.SubmanifoldConvolution(dimension, in_channels, nPlanesF[0], 3, False))

        self.layers_out = scn.Sequential(
            scn.BatchNormReLU(nPlanesF[0]),
            scn.OutputLayer(dimension))

        self.linear = nn.Linear(nPlanesF[0], 20)

        #**********************************************************************#

        def block(m, a, b):
            if residual_blocks: #ResNet style blocks
                m.add(scn.ConcatTable()
                      .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                      .add(scn.Sequential()
                        .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                        .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                        .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                        .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
                 ).add(scn.AddTable())
            else: #VGG style blocks
                m.add(scn.Sequential()
                     .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                     .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))
            operation = {'kernel':[1,1,1], 'stride':[1,1,1]}
            return operation

        def down(m, nPlane_in, nPlane_downed, scale):
          #print(f'down, scale={scale}, feature={nPlane_in}->{nPlane_downed}, kernel={self.down_kernels[scale]},stride={self.down_strides[scale]}')
          m.add(scn.Sequential()
                  .add(scn.BatchNormLeakyReLU(nPlane_in,leakiness=leakiness))
                  .add(scn.Convolution(dimension, nPlane_in, nPlane_downed,
                          self.down_kernels[scale], self.down_strides[scale], False)))
          operation = {'kernel':self.down_kernels[scale], 'stride':self.down_strides[scale]}
          return operation

        def up(m, nPlane_in, nPlane_uped, scale):
          #print(f'up, scale={scale}, feature={nPlane_in}->{nPlane_uped}, kernel={self.down_kernels[scale]}, stride={self.down_strides[scale]}')
          m.add( scn.BatchNormLeakyReLU(nPlane_in, leakiness=leakiness)).add(
                      scn.Deconvolution(dimension, nPlane_in, nPlane_uped,
                      self.down_kernels[scale], self.down_strides[scale], False))
          operation = {'kernel':self.down_kernels[scale], 'stride':self.down_strides[scale]}
          return operation


        scales_num = len(nPlanesF)
        m_downs = nn.ModuleList()
        m_shortcuts = nn.ModuleList()
        operations_down = []
        for k in range(scales_num):
            m = scn.Sequential()
            if k > 0:
              op = down(m, nPlanesF[k-1], nPlanesF[k], k-1)
              operations_down.append(op)
            for _ in range(reps):
              op = block(m, nPlanesF[k], nPlanesF[k])
              if k==0:
                operations_down.append(op)
            m_downs.append(m)

            m = scn.SubmanifoldConvolution(dimension, nPlanesF[k], nPlaneM, 1, False)
            m_shortcuts.append(m)

        ###
        m_ups = nn.ModuleList()
        m_mergeds = nn.ModuleList()
        operations_up = []
        for k in range(scales_num-1, 0, -1):
            m = scn.Sequential()
            op = up(m, nPlaneM, nPlaneM, k-1)
            m_ups.append(m)
            operations_up.append(op)

            m_mergeds.append(scn.SubmanifoldConvolution(dimension, nPlaneM, nPlaneM, 3, False))

            #m = scn.Sequential()
            #for i in range(reps):
            #    block(m, nPlanesF[k-1] * (1+int(self._merge=='cat') if i == 0 else 1), nPlanesF[-1])
            #m_ups_decoder.append(m)

        self.m_downs = m_downs
        self.m_shortcuts = m_shortcuts
        self.m_ups = m_ups
        self.m_mergeds = m_mergeds
        self.operations_down = operations_down
        self.operations_up = operations_up

    def forward(self, net0):
      if self._show: print(f'\nFPN net input: {net0[0].shape}')
      net1 = self.layers_in(net0)
      net_scales = self.forward_fpn(net1)

      #net_scales = [n.to_dict() for n in net_scales]
      return net_scales

    def forward_fpn(self, net):
      if self._show:
        print('input sparse format:')
        sparse_shape(net)

      scales_num = len(self.m_downs)
      downs = []
      #if self._show:    print('\ndowns:')
      for m in self.m_downs:
        net = m(net)
        #if self._show:  sparse_shape(net)
        downs.append(net)

      net = self.m_shortcuts[-1](net)
      ups = [net]
      #if self._show:    print('\nups:')
      fpn_scales_from_back = [scales_num-1-i for i in self.fpn_scales]
      fpn_scales_from_back.sort()
      for k in range(scales_num-1):
        if k >= max(fpn_scales_from_back):
          continue
        j = scales_num-1-k-1
        net = self.m_ups[k](net)
        #if self._show:  sparse_shape(net)
        shorcut = self.m_shortcuts[j]( downs[j] )
        net = scn.add_feature_planes([ net, shorcut ])
        #net = self.m_ups_decoder[k](net)
        #if self._show:  sparse_shape(net)
        ups.append(self.m_mergeds[k](net))

      fpn_maps = [ups[i] for i in fpn_scales_from_back]

      if self._show:
        receptive_field(self.operations_down)
        print('\n\nSparse FPN\n--------------------------------------------------')
        print(f'scale num: {scales_num}')
        print('downs:')
        for i in range(len(downs)):
          #if i!=0:
          #  print(f'\tKernel:{self.down_kernels[i-1]} stride:{self.down_strides[i-1]}', end='\t')
          #else:
          #  print('\tSubmanifoldConvolution \t\t', end='\t')
          op = self.operations_down[i]
          ke = op['kernel']
          st = op['stride']
          rf = op['rf']
          tmp = f' \tKernel:{ke}, Stride:{st}, Receptive:{rf}'
          sparse_shape(downs[i], pre=f'\t{i} ', post=tmp)

        print('\n\nups:')
        for i in range(len(ups)):
          #if i==0:
          #  print('\tIdentity of the last \t\t', end='\t')
          #else:
          #  print(f'\tKernel:{self.down_kernels[-i]} stride:{self.down_strides[-i]}', end='\t')
          op = self.operations_up[i]
          ke = op['kernel']
          st = op['stride']
          tmp = f' \tKernel:{ke}, Stride:{st}'
          sparse_shape(ups[i], post=tmp)

        print('\n\nFPN_Net out:')
        print(f'{fpn_scales_from_back} of ups')
        for t in fpn_maps:
          sparse_shape(t)
          sparse_real_size(t,'\t')
          print('\n')
        print('--------------------------------------------------\n\n')
      return fpn_maps


def receptive_field(operations, voxel_size_basic = None):
  '''
  https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
  '''
  n = len(operations)
  operations[0]['rf'] = np.array([1.0,1,1])
  jump = 1
  for i in range(1,n):
    op = operations[i]
    ke = np.array(op['kernel'])
    st = np.array(op['stride'])
    rf = operations[i-1]['rf'] + (ke-1)*jump
    operations[i]['rf'] = rf
    jump *= st

  if voxel_size_basic:
    for op in operations:
      op['rf'] *= voxel_size_basic

def sparse_shape(t, pre='\t', post=''):
  print(f'{pre}{t.features.shape}, {t.spatial_size}{post}')

def sparse_real_size(t,pre=''):
  loc = t.get_spatial_locations()
  loc_min = loc.min(0)[0]
  loc_max = loc.max(0)[0]
  print(f"{pre}min: {loc_min}, max: {loc_max}")

