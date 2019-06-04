class SceneSamples():
    '''
    '''
    err_scenes = ['01b05d5581c18177f6e8444097d89db4', '01c3dd293fc00701d2239e9e58e03967']

    # bad samples: not used in the training (1) BIM definition ambiguous
    bad_scenes = ['004e36a61e574321adc8da7b48c331f2', '00466151039216eb333369aa60ea3efe', '008969b6e13d18db3abc9d954cebe6a5', '0165e8534588c269219c9aafa9d888da']
    #               curved walls
    bad_scenes += ['019853e4742f679151c34f2732c33c16']
    bad_scenes += err_scenes

    # good samples
    good_samples_complex = [ '0058113bdc8bee5f387bb5ad316d7b28', '005f0859081006be329802f967623015', '007802a5b054a16a481a72bb3baca6a4','00922f91aa09dbdda3a74489ea0e21eb']
    #                                                           80a21c need cro pto view
    good_samples_angle = ['00602d3d932a8d5305234360a9d1e0ad', '0067620211b8e6459ff24ebe0780a21c']

    # hard exampels: (1)
    hard_samples_long_wall = ['00466151039216eb333369aa60ea3efe']
    hard_samples_close_walls = ['001ef7e63573bd8fecf933f10fa4491b', '01b1f23268db0f2801f4685a7e1563b9']
    hard_samples_notwall_butsimilar = ['0016652bf7b3ec278d54e0ef94476eb8']
    hard_samples_window_wall_close = ['01b8fe9faef3a608714e93be9dc9fac1']
    hard_samples_short_wall = ['01b8fe9faef3a608714e93be9dc9fac1']

    #---------------------------------------------------------------------------

    # very hard
    very_hard_wall_window_close = '0055398beb892233e0664d843eb451ca' # a lot of windows are almost same with wall

    # hard and error_prone scenes
    hard_id1 = '001ef7e63573bd8fecf933f10fa4491b'  # two very close walls can easily be merged as one incorrectly (very hard to detect)
    hard_id3 = '002f987c1663f188c75997593133c28f'  # very small angle walls, ambiguous in wall definition
    hard_id4 = '00466151039216eb333369aa60ea3efe'  # too long wall
    hard_id5 = '004e36a61e574321adc8da7b48c331f2'  # complicated and wall definitoin ambiguous

    # hard to parse, but fixed already
    parse_hard_id0 = '0058113bdc8bee5f387bb5ad316d7b28'  # a wall is broken by no intersection

    #
    scene_id0 = '31a69e882e51c7c5dfdc0da464c3c02d' # 68 walls
    scene_id1 = '8c033357d15373f4079b1cecef0e065a' # one level, with yaw!=0, one wall left and right has angle (31 final walls)

