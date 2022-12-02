import io
import base64
import json
from pathlib import Path
from PIL import Image

import numpy as np


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    img_arr = np.array(img_pil)
    return img_arr


def img_arr_to_b64(img_arr):
    img_pil = Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def img_data_to_pil(img_data):
    img_arr = img_b64_to_arr(img_data)
    return Image.fromarray(img_arr)


def img_pil_to_data(img_pil):
    img_arr = np.array(img_pil)
    img_b64 = img_arr_to_b64(img_arr)
    return img_b64.decode("utf-8")


def interpolate_json_files(json_files_list, out_path, times=3):
    for old_json_path in json_files_list:
        json_obj = open(old_json_path)
        annotation = json.load(json_obj)
        old_shapes = annotation["shapes"]
        old_img_data = annotation["imageData"]
        old_img_path = annotation["imagePath"]

        old_height = annotation["imageHeight"]
        old_width = annotation["imageWidth"]

        new_height = int(old_height * times)
        new_width = int(old_width * times)

        old_img_pil = img_data_to_pil(old_img_data)

        new_img_pil = old_img_pil.resize((new_width, new_height),
                                         Image.Resampling.BICUBIC)

        new_img_data = img_pil_to_data(new_img_pil)

        new_shapes = []
        for cell in old_shapes:
            old_poly = np.array(cell["points"])
            old_x, old_y = old_poly[:, 0], old_poly[:, 1]

            # Interpolating coordinates of labelme polygons
            new_x = old_x * (new_height / old_height)
            new_y = old_y * (new_width / old_width)

            new_poly = [[i, j] for i, j in zip(new_x, new_y)]
            cell["points"] = new_poly
            new_shapes.append(cell)

        new_img_path = out_path + "/" + str(old_img_path)
        new_json_path = out_path + "/" + str(old_json_path.name)

        annotation["imagePath"] = new_img_path
        annotation["imageData"] = new_img_data
        annotation["shapes"] = new_shapes

        annotation["imageHeight"] = new_height
        annotation["imageWidth"] = new_width

        new_img_pil.save(new_img_path)
        with open(new_json_path, "w") as handle:
            json.dump(annotation, handle, indent=2)


if __name__ == "__main__":
    p = r'C:\Raghava_local\datasets'

    json_files = [p for p in Path(p).rglob("*img.json")]

    interpolate_json_files(p)
