import h5py
import numpy as np


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add suffix
    if magnitude == 1:
        return '%.1f%s' % (num, 'K')
    elif magnitude == 2:
        return '%.1f%s' % (num, 'M')
    elif magnitude == 3:
        return '%.1f%s' % (num, 'B')
    else:
        return '%.1f' % num


def create_hdf5(images, masks, predicts, diffs, filename=None):
    compression = "gzip"
    with h5py.File(filename, "a") as h:
        # scores = h.create_dataset(name="events/avg_IOU",
        #                           data=(avg_score,),
        #                           dtype="float16")
        # inftime = h.create_dataset(name="events/inference_time",
        #                            data=(infr_time,),
        #                            dtype="float16")
        imset = h.create_dataset(name="events/image",
                                 data=np.array(images),
                                 shape=np.array(images).shape,
                                 dtype=np.array(images).dtype,
                                 compression=compression
                                 )
        imset.attrs.create("CLASS", "IMAGE", dtype="S6")
        imset.attrs.create("IMAGE_SUBCLASS", "IMAGE_GRAYSCALE", dtype="S16")
        imset.attrs.create("IMAGE_VERSION", "1.2", dtype="S4")
        imset.attrs.create("INTERLACE_MODE", "INTERLACE_PIXEL", dtype="S16")
        mskset = h.create_dataset(name="events/mask",
                                  data=np.array(masks),
                                  shape=np.array(masks).shape,
                                  dtype=np.array(masks).dtype,
                                  compression=compression
                                  )
        mskset.attrs.create("CLASS", "IMAGE", dtype="S6")
        mskset.attrs.create("IMAGE_SUBCLASS", "IMAGE_GRAYSCALE", dtype="S16")
        mskset.attrs.create("IMAGE_VERSION", "1.2", dtype="S4")
        mskset.attrs.create("INTERLACE_MODE", "INTERLACE_PIXEL", dtype="S16")
        predset = h.create_dataset(name="events/prediction",
                                   data=np.array(predicts),
                                   shape=np.array(predicts).shape,
                                   dtype=np.array(predicts).dtype,
                                   compression=compression
                                   )
        predset.attrs.create("CLASS", "IMAGE", dtype="S6")
        predset.attrs.create("IMAGE_SUBCLASS", "IMAGE_GRAYSCALE", dtype="S16")
        predset.attrs.create("IMAGE_VERSION", "1.2", dtype="S4")
        predset.attrs.create("INTERLACE_MODE", "INTERLACE_PIXEL", dtype="S16")
        diffset = h.create_dataset(name="events/msk-pred",
                                   data=np.array(diffs),
                                   shape=np.array(diffs).shape,
                                   dtype=np.array(diffs).dtype,
                                   compression=compression
                                   )
        diffset.attrs.create("CLASS", "IMAGE", dtype="S6")
        diffset.attrs.create("IMAGE_SUBCLASS", "IMAGE_GRAYSCALE", dtype="S16")
        diffset.attrs.create("IMAGE_VERSION", "1.2", dtype="S4")
        diffset.attrs.create("INTERLACE_MODE", "INTERLACE_PIXEL", dtype="S16")
