import numpy as np
import nibabel as nib
import glob
import cv2
import scipy.misc as scipy
import os
import shutil
from random import random, uniform, randint

import configs as cfgs

from albumentations import (
    Compose,
    OneOf,
    HorizontalFlip,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    Blur,
    MedianBlur,
    MotionBlur,
    Cutout,
    IAASharpen,
    IAAEmboss,
    IAAAdditiveGaussianNoise,
    IAAPiecewiseAffine,
    RandomGamma
)
'''
    aug = Compose([
        Compose([
            ElasticTransform(alpha=0.2, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, p=1.),
            GridDistortion(p=0.3),
            OpticalDistortion(p=0.3),
            IAAPiecewiseAffine(p=0.3)
        ], p=1.),
        IAASharpen(p=0.3, alpha=(0.2, 0.5), lightness=(0.5, 1.0) ),
        IAAPiecewiseAffine(scale=0.03, nb_rows=2, nb_cols=2, order=1, cval=0, mode='constant', p=0.5),
        IAAAdditiveGaussianNoise(loc=0, scale=(10, 30), per_channel=False, p=0.3),
    ])

    aug = ElasticTransform(alpha=1., sigma=50, alpha_affine=10, interpolation=1, border_mode=0, p=1.)
    aug = GridDistortion(num_steps=50, distort_limit=0.2, interpolation=1, border_mode=0, p=1.)
    aug = OpticalDistortion(distort_limit=0, shift_limit=25, interpolation=1, border_mode=0, p=1.)
    aug = IAAPiecewiseAffine(scale=(0.01, 0.03), nb_rows=3, nb_cols=3, order=1, cval=0, mode='constant', p=1.)
    aug = IAASharpen(alpha=(0.1, 0.3), lightness=(1., 1.), p=1.)
    aug = IAAEmboss(alpha=(0.25, 0.5), strength=(0.2, 0.3), p=1.)
'''
import imgaug as ia
from imgaug import augmenters as iaa

def data_augmentation(image, roi):
    if to_apply(p=0.5):
        image = aorta_boot(image, roi, min_boot=0.25, max_boot=0.60)

    aug = Compose([
        OneOf([
            ElasticTransform(alpha=1., sigma=50, alpha_affine=10, interpolation=1, border_mode=0, p=0.25),
            GridDistortion(num_steps=50, distort_limit=0.2, interpolation=1, border_mode=0, p=0.25),
            IAAPiecewiseAffine(scale=(0.01, 0.03), nb_rows=3, nb_cols=3, order=1, cval=0, mode='constant', p=0.25),
            OpticalDistortion(distort_limit=0, shift_limit=25, interpolation=1, border_mode=0, p=0.25)
        ], p=0.3),
        IAAAdditiveGaussianNoise(loc=0, scale=(10, 25), per_channel=False, p=0.3)
    ], p=0.7)
    
    
    augmented = aug(image=image, mask=roi)

    return augmented['image'], augmented['mask']

def aorta_boot(image, roi, min_boot=0.3, max_boot=0.5):
    max_val = np.amax(np.where(roi==1., image, 0) )
    boot_percent = uniform(min_boot, max_boot)
    
    if max_val > 1.0:
        return np.where(roi==1., image+((255.0-image)*boot_percent), image)
    else:
        return np.where(roi==1., image+((1.-image)*boot_percent), image)

def to_apply(p=0.5):
    if random() <= p:
        return True
    else:
        return False

def open_im_roi(path):
    if path.find(".npz") > 0:
        return np.load(path)['arr_0']
    else:
        return nib.load(path).get_fdata().swapaxes(0, 1)

def nityi_cap(imgs, num1=300.0, num2=300.0):
    num1 = float(num1)
    num2 = float(num2)
    d = imgs.astype(np.float32)
    d = d + num1
    d = np.where(d < 0.0, 0.0, d)
    d = (d / num1+num2)
    d = np.where(d > 1.0, 1.0, d)
    return d

def process_data(data):
    #data -= np.amin(data)
    #data /= np.amax(data)
    data = np.clip(np.fabs(data), -np.inf, np.inf)

    num = np.amax(data)-np.amin(data)

    if num != 0.0:
        return (data-np.amin(data))*(1.-0.)/num
    else:
        return data

def process_roi(roi):
    roi = roi.astype(np.bool)
    if cfgs.n_classes > 1:
        tmp_roi = np.zeros([cfgs.train_size_ny, cfgs.train_size_nx, cfgs.n_classes], dtype=np.bool)
        tmp_roi[..., 0] = ~roi
        tmp_roi[..., 1] = roi
        return tmp_roi
    
    return roi

def flat_data(data, n_ch):
    return data.reshape(cfgs.train_size_ny, cfgs.train_size_nx, n_ch)

def all_path(path):
    return sorted(glob.glob(path))

def tail_word(path):
    return os.path.basename(path)

def find_best_path(all_get):
    c_max = -1
    c_path = ""
    num_epoch = 0
    for path in(all_get):
        score = float(path[path.rfind("_")+1:path.rfind(".")])
        if score >= c_max:
            c_max = score
            c_path = path
            num_epoch = int(path[path.find("epoch_")+6:path.find("-")])

    return num_epoch, c_path

def set_check_pred(images, rois, arr_select):
    for i in range(len(arr_select)):
        cfgs.check_images.append(images[arr_select[i]])
        cfgs.check_rois.append(rois[arr_select[i]])

def crop_size(image, ny, nx):
    (h, w) = image.shape

    t = int((h - ny)/2)
    b = int((h + ny)/2)
    l = int((w - nx)/2)
    r = int((w + nx)/2)
    
    return image[t:b, l:r]

def pad_size(image, ny, nx):
    (h, w) = image.shape

    img = np.zeros([ny, nx], dtype=np.float32)
    img[0:h, 0:w] = image

    return img

def cal_step(image):
    (h, w) = image.shape
    step_h = h/cfgs.divide_stride
    step_w = w/cfgs.divide_stride

    if step_w > int(step_w):
        step_w += 1
    if step_h > int(step_h):
        step_h += 1

    step_h = int(step_h)
    step_w = int(step_w)

    return step_h, step_w

def ori_position(image, num_best_piece):
    step_h, step_w = cal_step(image)
    
    line = int(num_best_piece / step_w)
    t = line * cfgs.divide_stride
    b = t + cfgs.train_size_ny
    l = (num_best_piece % step_w) * cfgs.divide_stride
    r = l + cfgs.train_size_nx
    
    return t, b, l, r

def pred_in_ori_position(image, b, r):

    pred_b = cfgs.train_size_ny
    pred_r = cfgs.train_size_nx
    
    if b > image.shape[0]:
        pred_b -= (b - image.shape[0])
        b = image.shape[0]
        
    if r > image.shape[1]:
        pred_r -= (r - image.shape[1])
        r = image.shape[1]

    return pred_b, pred_r

def to_channel_last(arr):
    last = arr.ndim - 1
    return arr.swapaxes(0, last)

def divide_image(image):
    step_h, step_w = cal_step(image)

    divide_images = np.zeros([step_h*step_w, cfgs.train_size_ny, cfgs.train_size_nx], dtype=np.float32)

    for y in range(step_h):
        for x in range(step_w):

            t = y*cfgs.divide_stride
            b = t + cfgs.train_size_ny
            l = x*cfgs.divide_stride
            r = l + cfgs.train_size_nx

            divide_images[x+(y*step_w), ] = pad_size(image[t:b, l:r], cfgs.train_size_ny, cfgs.train_size_nx)
            #save_img(pad_size(image[t:b, l:r], cfgs.train_size_ny, cfgs.train_size_nx), cfgs.outpath_pred+"pic/", "img{:}".format(x+(y*step_w)))

    return divide_images

def find_best_piece(divide_images):
    all_sum_each = cal_sum_each(divide_images)
    return index_of_best(all_sum_each)

def index_of_best(all_sum_each):
    m = max(all_sum_each)
    indexs = [num for num, x in enumerate(all_sum_each) if x == m]
    return indexs[int(len(indexs)/2)]

def cal_sum_each(divide_images):
    all_sum = []
    for i in range(divide_images.shape[0]):
        all_sum.append(np.sum(divide_images[i, ]) )

    return all_sum

def auto_fill_bs(text):
	if text[-1] != "/":
		text+="/"
	return text

def mask_on_image(image, pred, prob=0.5):
    img = to_rgb(image)
    mask_channel = img[..., 1]
    img[..., 1] = np.where(pred > prob, 0.7*mask_channel+0.5, mask_channel)

    return img

def to_rgb(image):
    img = np.atleast_3d(image)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img *= 255
    return img

def combine_img_prediction(image, y_pred):
    img = np.concatenate((to_rgb(image), 
                          to_rgb(y_pred),
                          to_rgb( ((y_pred > 0.5).astype(np.float32)) )) , axis=1)
    return img


def combine_img_prediction_gt(image, y_true, y_pred):
    img = np.concatenate((to_rgb(image), 
                          to_rgb(y_true), 
                          to_rgb(y_pred),
                          to_rgb( ((y_pred > 0.5).astype(np.float32)) )) , axis=1)
    return img

def save_img(image, outpath, name, dtype=".png"):
    scipy.imsave(outpath+name+dtype, image)

def to_nii_gz(in_arr, name, outpath):
    arr = in_arr
    data = nib.Nifti1Image(arr, np.eye(4))
    nib.save(data, outpath+name)

def to_npz(in_arr, name, outpath):
    arr = in_arr
    np.savez(outpath+name, arr)
