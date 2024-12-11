# trying some examples from taichi docs

import taichi as ti

vertices         = ti.Vector.field(2, ti.f32, shape=200)
vertices_3d      = ti.Vector.field(3, ti.f32, shape=200)
indices          = ti.field(ti.i32, shape=200 * 3)
normals          = ti.Vector.field(3, ti.f32, shape=200)
per_vertex_color = ti.Vector.field(3, ti.f32, shape=200)

color  = (0.5, 0.5, 0.5)


window = ti.ui.Window("Test for GUI", res=(512, 512))
gui = window.get_gui()
value = 0
color = (1.0, 1.0, 1.0)
with gui.sub_window("Sub Window", x=10, y=10, width=300, height=100):
    gui.text("text")
    is_clicked = gui.button("name")
    value = gui.slider_float("name1", value, minimum=0, maximum=100)
    color = gui.color_edit_3("name2", color)

while window.running:
    window.show()