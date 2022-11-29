import base64
import io
import json
from PIL import Image
import warnings

import numpy as np
from scipy import interpolate
from skimage.measure import grid_points_in_poly


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


def json_to_mask(json_list, correction=0.4):
    images, masks = [], []
    for json_file in json_list:
        json_obj = open(json_file)
        annotation = json.load(json_obj)
        cells = annotation["shapes"]
        img = annotation["imageData"]
        if len(cells) != 0 and img is not None:
            image = img_b64_to_arr(img)
            temp_mask = []
            for cell in cells:
                poly = np.array(cell["points"])
                x, y = poly[:, 0], poly[:, 1]
                x, y = x-correction, y-correction
                xy = np.array([[p[1], p[0]] for p in zip(x, y)])
                assert len(xy) > 2, "Polygon must have points more than 2"
                mask = grid_points_in_poly(image.shape[:2], xy)
                mask = np.array(mask, dtype='bool')
                temp_mask.append(mask)
            mask = np.sum(temp_mask, axis=0, dtype="float32")
            images.append(image)
            masks.append(mask)
    return images, masks


def get_labeme_shape(xi, yi, cell_lbl):
    shape = np.array((xi, yi)).T
    cell = {
        "label": cell_lbl,
        "points": shape.tolist(),
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
    }
    return cell


def create_json(img, cnts, cell_labels, img_file_path,
                only_valid=False, x_off=0.5, y_off=0.5):
    img_height, img_width = img.shape
    shapes_list = []

    for cnt, cell_lbl in zip(cnts, cell_labels):
        # Computed contours (measure.find_contours) have integer type
        # coordinates but when we create a labelme file (JSON), labelme
        # converts these coordinates into float type. Due to this, there
        # is a shift in labelme contour. To avoid this, offset value is
        # added to the coordinates.
        x, y = cnt[:, 1] + x_off, cnt[:, 0] + y_off

        # Interpolate x, y coords to find the smooth curve
        warnings.filterwarnings("error")
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            tck = interpolate.splprep([x, y], k=2, s=0.4, per=True)[0]

        # Find uniformly distributed limited number of coords
        if len(x) < 50:
            num_points = int(len(x) / 3)
        else:
            num_points = int(len(x) / 5)

        xi, yi = interpolate.splev(np.linspace(0, 1, num_points), tck)

        # Polygon should have more than 3 coordinates
        if len(xi) > 3:
            # If the detected contour is 2-pixels away from image boundary
            # then it is treated as valid blood cell. But, any of the detected
            # contour close to image boundary then it is treated as invalid
            # blood cell. Sometimes due to interpolation, x-coordinates are
            # computed beyond the image boundary.  So, lines (xi[xi < 0] = 0 &
            # xi[xi > 249] = 250) fix them to be on the boundary.

            if only_valid:
                # Consider all cells with in the frame
                if np.all(xi < 248) and np.all(xi > 2):
                    shape = get_labeme_shape(xi, yi, cell_lbl)
                    shapes_list.append(shape)
            else:
                # Include cells at the boundary of the frame with
                # ``_invalid`` string attached to the label
                if np.any(xi > 248) or np.any(xi < 2):
                    xi[xi < 2] = 0.0
                    xi[xi > 248] = 249.0
                    cell_lbl = cell_lbl + "_invalid"
                shape = get_labeme_shape(xi, yi, cell_lbl)
                shapes_list.append(shape)

    image_data = img_arr_to_b64(img).decode("utf-8")
    json_file = str(img_file_path).replace(".png", ".json")

    # Create labelme json object
    json_dict = {"version": "5.0.1",
                 "flags": {},
                 "shapes": shapes_list,
                 "imagePath": img_file_path.name,
                 "imageData": image_data,
                 "imageHeight": img_height,
                 "imageWidth": img_width}

    # Save image_bg
    img_pil = Image.fromarray(img)
    img_pil.save(str(img_file_path))

    # Save json file
    with open(json_file, "w") as handle:
        json.dump(json_dict, handle, indent=2)


def get_img_bg_from_rtdc_and_json_file(rtdc_dataset, json_files):
    frames = np.array(rtdc_dataset['frame'])
    image_bgs = rtdc_dataset['image_bg']
    for jfile in json_files:
        frm = float((str(jfile).split('frm_')[1]).split('_idx')[0])
        idx = np.where(frames == frm)
        img_bg = image_bgs[idx][0]
        img_bg_file_name = str(jfile).replace('.json', '_img_bg.png')
        img_bg_pil = Image.fromarray(img_bg)
        img_bg_pil.save(img_bg_file_name)
