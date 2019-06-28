class SceneSamples():
    '''
    '''
    #---------------------------------------------------------------------------
    paper_samples = ['00602d3d932a8d5305234360a9d1e0ad', '0058113bdc8bee5f387bb5ad316d7b28', '0055398beb892233e0664d843eb451ca']

    #---------------------------------------------------------------------------
    err_scenes = []

    #---------------------------------------------------------------------------

    bad_scenes_curved_walls = ['020179798688014a482f483e1a5debe5', '019853e4742f679151c34f2732c33c16']

    # bad samples: not used in the training (1) BIM definition ambiguous
    bad_scenes_BIM_ambiguous = ['004e36a61e574321adc8da7b48c331f2', '00466151039216eb333369aa60ea3efe', '008969b6e13d18db3abc9d954cebe6a5', '0165e8534588c269219c9aafa9d888da']
    bad_scenes_raw_bad = ['0320272d1b3c30e2d9f897ff917cef15']
    # 0705e25aa6919af45268133bd2d98b65: no wall for window
    bad_scenes_cannot_parse_no_wall_for_window = ['13304f20f6327c21aa285069efb03ca1', '0705e25aa6919af45268133bd2d98b65', '142686fa469dda10dae66065be7961ef']
    bad_scenes_cannot_parse = ['032e05d444b03cc1c80c0700ad4238b1', '0382e82fab999376ef880fcff345090d'] + \
      ['100bcb702b28198108369345bf26f302', '1102fd6dc8702f1cd0f1f21508cce0bb', '110385ba3254a1816cc67a1b78243823'] +\
      ['14535bf081bd5ad2072683b43c8f0fd8', '14ab942f5f42112c1b2afa341b2b7522', '1515923b28f1cd8b101cc1f74358bb92'] +\
      bad_scenes_cannot_parse_no_wall_for_window

    bad_scenes = bad_scenes_curved_walls + bad_scenes_BIM_ambiguous + bad_scenes_raw_bad + err_scenes + bad_scenes_cannot_parse

    bad_scenes_submanifold_bs1_nan = ['0005b50577f5871e1c0bb7a687f6cbc3']

    #---------------------------------------------------------------------------

    # good samples
    good_samples_complex = [ '0058113bdc8bee5f387bb5ad316d7b28', '005f0859081006be329802f967623015', '007802a5b054a16a481a72bb3baca6a4','00922f91aa09dbdda3a74489ea0e21eb']
    #                                                           80a21c need cro pto view
    good_samples_angle = ['00602d3d932a8d5305234360a9d1e0ad', '0067620211b8e6459ff24ebe0780a21c', '02164f84a9e7321f3071b2214df8c738']

    # hard exampels: (1)
    hard_samples_long_wall = ['00466151039216eb333369aa60ea3efe']
    hard_samples_close_walls = ['001ef7e63573bd8fecf933f10fa4491b', '01b1f23268db0f2801f4685a7e1563b9']
    hard_samples_notwall_butsimilar = ['0016652bf7b3ec278d54e0ef94476eb8']
    hard_samples_window_wall_close = ['01b8fe9faef3a608714e93be9dc9fac1']
    hard_samples_short_wall = ['01b8fe9faef3a608714e93be9dc9fac1']


    # very hard
    very_hard_wall_window_close = ['0055398beb892233e0664d843eb451ca'] # a lot of windows are almost same with wall
    very_hard_windows_close = ['001e3c88f922f42b5a3f546def6eb83f']





    #---------------------------------------------------------------------------

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

