# xyz May 2019

def read_obj(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    n = len(lines)

    del_blocks = []
    find_end = 0
    new_lines = []
    for i in range(n):
        if find_end == 0 and  lines[i][0:9] == 'g Ceiling':
            del_start = i
            find_end = 1
        if find_end==1 and lines[i][0] !='g':
            find_end = 2
        if find_end == 2 and lines[i][0] =='g':
            del_end = i-1
            find_end = 0
            del_blocks.append( (del_start, del_end) )

        if find_end == 2 and lines[i][0] == 'f':
            pass
        else:
            new_lines.append(lines[i])

    print(del_blocks)

    n_new = len(new_lines)
    print(f'{n} -> {n_new}')

    new_fn = fn.replace('house.obj', 'no_ceiling_house.obj')
    with open(new_fn, 'w') as f:
        for l in new_lines:
            f.write(l)
        print(f'write ok :\n {new_fn}')

if __name__ == '__main__':
    folder = '/home/z/SUNCG/suncg_v1/parsed'
    house_name = '31a69e882e51c7c5dfdc0da464c3c02d'
    house_name = '8c033357d15373f4079b1cecef0e065a'
    fn = f'{folder}/{house_name}/house.obj'
    read_obj(fn)
