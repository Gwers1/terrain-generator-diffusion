import torch
import base64
import bpy
import tempfile
import numpy as np

from os.path import basename
from SimpleUnet import SimpleUnet
from sampling import sample_plot_image, sample_image
from torchvision import transforms
from io import BytesIO

PATH = "model/modelModel50.pt"
model = SimpleUnet()
model.load_state_dict(torch.load(PATH, weights_only=False))
model.eval()
model.to("cuda")

def heightmap_generation(num):
    imgTensor = sample_image(model)
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    if len(imgTensor.shape) == 4:
        imgTensor = imgTensor[0, :, :, :] 

    img = reverse_transforms(imgTensor.detach().cpu())
    
    name = 'heightmaps/heightmapGen0' + str(num) + '.png'
    img.save(name, format = 'png')

    return img
    

def PILtoBase64(img):
    heightmap_buffer = BytesIO()
    img.save(heightmap_buffer, format = 'png')
    heightmap_bytes = heightmap_buffer.getvalue()
    heightmap_base64 = base64.b64encode(heightmap_bytes).decode('utf-8')

    return heightmap_base64

def clear_page():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)

def create_plane():
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0), enter_editmode=True)
    plane_obj = bpy.data.objects[0]
    plane_obj.name = 'MyPlane'
    return plane_obj

def subdivide(num_cuts):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=num_cuts)
    bpy.ops.object.mode_set(mode='OBJECT')

def add_modifiers(plane_obj, heightmap_base64):

    subsurf = plane_obj.modifiers.new(name='Subdivision Surface', type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 5
    subsurf.subdivision_type = 'SIMPLE'

    mod = plane_obj.modifiers.new(name='Displace', type='DISPLACE')
    tex = bpy.data.textures.new(name='Hmap', type='IMAGE')

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        hmap_bytes = base64.b64decode(heightmap_base64.encode('utf-8'))
        f.write(hmap_bytes)
        temp_filepath = f.name

    tex.image = bpy.data.images.load(temp_filepath)
    mod.texture = tex
    mod.mid_level = 0.0
    mod.strength = 0.5

def render_blender(heightmap_base64, num, buffer=None, name="Terrain"):
    clear_page()
    plane_obj = create_plane()
    subdivide(100)
    add_modifiers(plane_obj, heightmap_base64)
    file_path = "terrains/terrainGen0" + str(num) + ".fbx"
    bpy.ops.export_scene.fbx(filepath=file_path, use_selection=True)


for x in range(978):
    heightmap_img = heightmap_generation(x)
    #heightmap_base64 = PILtoBase64(heightmap_img)
    #render_blender(heightmap_base64, x)

# heightmap_img = heightmap_generation()
# heightmap_base64 = PILtoBase64(heightmap_img)
# render_blender(heightmap_base64)