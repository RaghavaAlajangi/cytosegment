import base64
import io
import json
from pathlib import Path
from PIL import Image
import warnings

import numpy as np
from scipy.interpolate import splprep, splev
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


def json_to_mask(json_list, interpolate_rate=20):
    masks = []
    for json_file in json_list:
        json_obj = open(json_file)
        annotation = json.load(json_obj)
        cells = annotation["shapes"]
        img = annotation["imageData"]
        inter_height = annotation["imageHeight"]
        inter_width = annotation["imageWidth"]
        org_height = int(inter_height/interpolate_rate)
        org_width = int(inter_width / interpolate_rate)
        if len(cells) != 0 and img is not None:
            image = img_b64_to_arr(img)
            height, width = image.shape[:2]
            temp_masks = []
            for cell in cells:
                poly = np.array(cell["points"])
                x, y = poly[:, 0], poly[:, 1]
                # Eliminate the labelme offset at boundary while creating mask
                if np.any(x > (width - 2)):
                    x[x > (width - 2)] = width
                poly = np.array([[p[1], p[0]] for p in zip(x, y)])
                assert len(poly) > 2, "Polygon must have points more than 2"
                temp_mask = grid_points_in_poly(image.shape[:2], poly)
                temp_masks.append(temp_mask.astype('bool'))
            # Combine individual masks by applying logical_or
            mask = np.logical_or.reduce(np.array(temp_masks))
            # Down scale the mask to original size
            mask_pil = Image.fromarray(mask)
            mask_out = mask_pil.resize((org_width, org_height),
                                       Image.Resampling.NEAREST)
            mask_out = np.array(mask_out).astype("bool")
            masks.append(mask_out)
    return masks


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


def create_json(image, contours, cell_labels, img_file_path,
                only_valid=False, correction=0.5, interpolate_rate=20):
    old_height, old_width = image.shape[:2]

    new_height = int(old_height * interpolate_rate)
    new_width = int(old_width * interpolate_rate)

    old_img_pil = Image.fromarray(image)
    # Interpolate image to new shape
    new_img_pil = old_img_pil.resize((new_width, new_height),
                                     Image.Resampling.NEAREST)

    new_img_arr = np.array(new_img_pil)
    image_data = img_arr_to_b64(new_img_arr).decode("utf-8")

    shapes_list = []

    for cnt, cell_lbl in zip(contours, cell_labels):
        # Computed contours (measure.find_contours) have integer type
        # coordinates but when we create a labelme file (JSON), labelme
        # converts these coordinates into float64 type. Due to this, there
        # is a shift in labelme contour. To avoid this, offset value is
        # added to the coordinates.
        old_x, old_y = cnt[:, 1] + correction, cnt[:, 0] + correction

        # Interpolate x, y cords to find the smooth curve
        warnings.filterwarnings("error")
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            tck = splprep([old_x, old_y], k=2, s=0.4, per=True)[0]

        # Find uniformly distributed limited number of cords
        if len(old_x) < 50:
            num_points = int(len(old_x) / 2)
        else:
            num_points = int(len(old_x) / 3)

        ref = np.linspace(0, 1, num_points)
        inter_x, inter_y = splev(ref, tck)

        # Interpolating coordinates of labelme polygons
        new_x = inter_x * (new_width / old_width)
        new_y = inter_y * (new_height / old_height)

        # Polygon should have more than 3 coordinates
        if len(new_x) > 3:
            # If the detected contour is 2-pixels away from image boundary
            # then it is treated as valid blood cell. But, any of the detected
            # contour close to image boundary then it is treated as invalid
            # blood cell. Sometimes due to interpolation, x-coordinates are
            # computed beyond the image boundary.  So, below lines fix them
            # to be on the boundary.
            if only_valid:
                # Including all cells within the frame as valid cells
                if np.all(new_x < (new_width - 2)) and np.all(new_x > 2):
                    shape = get_labeme_shape(new_x, new_y, cell_lbl)
                    shapes_list.append(shape)
            else:
                # Including cells at the boundary of the frame with
                # ``_invalid`` string attached to the label
                if np.any(new_x > (new_width - 2)) or np.any(new_x < 2):
                    new_x[new_x < 2] = 0.0
                    new_x[new_x > (new_width - 2)] = float(new_width)
                    cell_lbl = cell_lbl + "_invalid"
                shape = get_labeme_shape(new_x, new_y, cell_lbl)
                shapes_list.append(shape)

    # Create paths for interpolated json and image files to be saved
    json_path = str(img_file_path).replace(".png", "_interpolated.json")
    new_img_path = str(img_file_path).replace(".png", "_interpolated.png")

    # Save image
    new_img_pil.save(str(new_img_path))

    # Create labelme json object
    json_dict = {"version": "5.0.1",
                 "flags": {},
                 "shapes": shapes_list,
                 "imagePath": Path(new_img_path).name,
                 "imageData": image_data,
                 "imageHeight": new_height,
                 "imageWidth": new_width}

    # Save json file
    with open(json_path, "w") as handle:
        json.dump(json_dict, handle, indent=2)


def get_img_bg_from_rtdc_and_json_file(rtdc_dataset, json_files):
    frames = np.array(rtdc_dataset['frame'])
    image_bgs = rtdc_dataset['image_bg']
    for json_file in json_files:
        frm = float((str(json_file).split('frm_')[1]).split('_idx')[0])
        idx = np.where(frames == frm)
        img_bg = image_bgs[idx][0]
        img_bg_file_name = str(json_file).replace('.json', '_img_bg.png')
        img_bg_pil = Image.fromarray(img_bg)
        img_bg_pil.save(img_bg_file_name)
