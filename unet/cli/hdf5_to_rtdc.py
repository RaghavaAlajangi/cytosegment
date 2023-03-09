import dclab
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


def get_rtdc_data(org_imgs, org_img_bgs, org_msks):
    frm_img = []
    frm_img_bg = []
    frm_msk = []
    frm = []
    for n, (img, ibg, msk) in enumerate(zip(org_imgs, org_img_bgs, org_msks)):
        msk = msk.astype('int8')
        segm, num = measure.label(msk, background=0, return_num=True)
        for jj in range(num):
            mask_jj = segm == jj + 1
            frm_img.append(img)
            frm_img_bg.append(ibg)
            frm_msk.append(mask_jj)
            frm.append(n)
    return frm_img, frm_img_bg, frm_msk, frm


def hdf5_to_rtdc(hdf5_file_path):
    hdf5_data = h5py.File(hdf5_file_path)
    hdf5_imgs = hdf5_data['events']['image']
    hdf5_img_bgs = hdf5_data['events']['image_bg']
    hdf5_msks = hdf5_data['events']['mask']

    hdf5_msks = np.array(hdf5_msks).astype('bool')

    img, img_bg, mask, frame = get_rtdc_data(hdf5_imgs, hdf5_img_bgs, hdf5_msks)

    new_rtdc_file_path = str(hdf5_file_path).replace('.hdf5', '.rtdc')
    with dclab.RTDCWriter(new_rtdc_file_path, mode="reset") as hw:
        hw.store_metadata({"experiment": {"sample": "my sample",
                                          "run index": 1}})
        hw.store_feature("image", img)
        hw.store_feature("image_bg", img_bg)
        hw.store_feature("mask", mask)
        hw.store_feature("frame", frame)


d = dclab.new_dataset(
    r"U:\Members\Raghava\00_SemanticSegmentation\UNET_annotations\old_train_data.rtdc")

print(d.features)

for i in range(4):
    plt.subplot(311)
    plt.imshow(d['image'][i], 'gray')
    plt.subplot(312)
    plt.imshow(d['image_bg'][i], 'gray')
    plt.subplot(313)
    plt.imshow(d['mask'][i], 'gray')
    plt.show()
