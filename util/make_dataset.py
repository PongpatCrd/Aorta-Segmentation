from util.data_util import all_path, open_im_roi, save_img, tail_word, crop_size
import configs as cfgs
import os
import shutil
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def make_dataset(num_val=0):
    logging.info("Creating dataset ...")
    gen_dataset(num_val)
    logging.info("All Completed!")

def gen_dataset(num_val):
    path_train_images = all_path(cfgs.inpath_imgs_md+"*")
    path_train_rois = all_path(cfgs.inpath_rois_md+"*")

    path_val_images = path_train_images[:num_val]
    path_val_rois = path_train_rois[:num_val]
    path_train_images = path_train_images[num_val:]
    path_train_rois = path_train_rois[num_val:]

    for i in range(len(path_train_images)):
        tmp_open_images = open_im_roi(path_train_images[i])
        tmp_open_rois = open_im_roi(path_train_rois[i])

        if path_train_images[i].rfind(".nii") > 0:
            tmp_open_images = dt.nityi_cap(tmp_open_images, 300, 300)

        images, rois = get_interest_slices(tmp_open_images, 
            tmp_open_rois, 
            cfgs.size_ny_md, 
            cfgs.size_nx_md,
            get_all=False)

        name = tail_word(path_train_images[i])
        name = name[:name.rfind(".")]

        for j in range(images.shape[2]-1):
            save_img(images[:,:,j], cfgs.path_train_images, name + "-{:03}".format(j+1))
            save_img(rois[:,:,j], cfgs.path_train_rois, name + "_roi-{:03}".format(j+1))
        logging.info("Complete {:}".format(i+1))

    for i in range(len(path_val_images)):
        tmp_open_images = open_im_roi(path_val_images[i])
        tmp_open_rois = open_im_roi(path_val_rois[i])

        if path_val_images[i].rfind(".nii") > 0:
            tmp_open_images = dt.nityi_cap(tmp_open_images, 300, 300)

        val_list(path_val_images[i])
        val_list(path_val_rois[i])

        images, rois = get_interest_slices(tmp_open_images, 
            tmp_open_rois, 
            cfgs.size_ny_md, 
            cfgs.size_nx_md, 
            get_all=False)

        name = tail_word(path_val_images[i])
        name = name[:name.rfind(".")]

        for j in range(images.shape[2]-1):
            save_img(images[:,:,j], cfgs.path_val_images, name + "-{:03}".format(j+1))
            save_img(rois[:,:,j], cfgs.path_val_rois, name + "_roi-{:03}".format(j+1))
        logging.info("Complete {:}".format(i+1+len(path_train_images)))

def _get_interest_marks(rois):
    interest_marks = []
    for i in range(rois.shape[2]):
        if rois[:, :, i].any() == 1.:
            interest_marks.append(i)

    return interest_marks

def get_interest_slices(images, rois, ny, nx, get_all):
    (h, w, n) = images.shape
    
    if ny == 0:
        ny = h

    if nx == 0:
        nx = w

    if get_all is False:
        marks = _get_interest_marks(rois)
    else:
        marks = np.arange(images.shape[2])

    interest_slices = np.zeros([ny, nx, len(marks)], dtype=np.float32)
    interest_rois = np.zeros([ny, nx, len(marks)], dtype=np.float32)

    for i in range(len(marks)):
        interest_slices[:, :, i] = crop_size(images[..., marks[i]], ny, nx)
        interest_rois[:, :, i] = crop_size(rois[..., marks[i]], ny, nx)

    return interest_slices, interest_rois

def val_list(path):
    filepath = cfgs.base_data_path + "val_list.txt"
    if os.path.isfile(filepath) is False:
        f = open(filepath, 'w')
        f.write(path+"\n")
    else:
        f = open(filepath, 'r')
        check = f.readlines()
        f.close()
        same = False
        for c in (check):
            if c.find(tail_word(path)) > 0:
                same=True
                break
        if same is False:
            f = open(filepath, 'a')
            f.write(path+"\n")
    f.close()
