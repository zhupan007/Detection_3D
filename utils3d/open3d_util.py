# 3 June
import numpy as np
import os,sys
import open3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def draw_cus(models):
    open3d.draw_geometries(models)
    return

    #open3d.visualization.RenderOption.line_width=5
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    for m in models:
        vis.add_geometry(m)
    #vis.get_render_option().line_width = 3
    vis.get_render_option().load_from_json(f"{BASE_DIR}/renderoption.json")
    vis.run()
    #print(f'point size: {vis.get_render_option().point_size}')
    #print(f'line width: {vis.get_render_option().line_width}')
    vis.destroy_window()
    pass

