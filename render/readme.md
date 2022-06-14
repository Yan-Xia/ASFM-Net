This directory contains code that generates partial point clouds from obj models. To use it:

1. Install Blender 2.79.
2. Create a model list. Each line of the model list should be in the format [model_id].
3. run normalize_one.py to normalize the obj models.
4. Run `blender -b -P render_depth.py [model directory] [model list] [output directory] [num scans per model]` to render the depth images. The images will be stored in OpenEXR format.
5. Run `python3 process_exr.py [model list] [intrinsics file] [output directory] [num scans per model]` to convert the `.exr` files into 16 bit PNG depth images and point clouds in the model's coordinate frame.