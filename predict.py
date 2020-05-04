import os
import shutil
import numpy as np
import configs as cfgs
import logging
import util.data_util as dt

from util.math_util import get_metrics, get_loss

from keras.models import load_model
from util.trainer import set_custom_object

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def main(in_path, out_path):
    best_model = dt.all_path(cfgs.best_model_path+"*")
    num_epoch, best_model = dt.find_best_path(best_model)
    logging.info("Load model..")
    loss = get_loss()
    metrics = get_metrics()
    set_custom_object(loss, metrics)
    model = load_model(best_model)
    logging.info("epoch: {:} path: {:}".format(num_epoch, best_model))

    imgs = dt.open_im_roi(in_path)

    if in_path.rfind(".nii") > 0:
        imgs = dt.nityi_cap(imgs, 300, 300)

    step_h, step_w = dt.cal_step(imgs[...,0]) 
    all_pred = []
    all_sum_each = []

    for i in range(imgs.shape[2]):
        x = dt.divide_image(imgs[..., i])
        pred = model.predict(x.reshape(x.shape[0], x.shape[2], x.shape[1], 1), cfgs.predict_batch_size)
        pred = (dt.process_data(pred[...,1]) > cfgs.threshold).astype(np.float32)
        all_pred.append(pred)
        cal_sum = dt.cal_sum_each(pred)

        if i == 0:
            for c in(cal_sum):
                all_sum_each.append(c)
        else:
            for c in range(len(cal_sum)):
                all_sum_each[c] += cal_sum[c]

    best_piece = dt.index_of_best(all_sum_each)
    t, b, l, r = dt.ori_position(imgs[..., 0], best_piece)
    pred_b, pred_r = dt.pred_in_ori_position(imgs[..., 0], b, r)
    final_arr = np.zeros([imgs.shape[0], imgs.shape[1], imgs.shape[2]])

    for j in range(len(all_pred)):
        final_arr[t:b, l:r, j] = all_pred[j][best_piece, :pred_b, :pred_r]

    name = dt.tail_word(in_path)[:dt.tail_word(in_path).find(".")] + cfgs.output_type
    
    if cfgs.output_type == ".npz":
        dt.to_npz(final_arr, name, out_path)
    elif cfgs.output_type == ".nii.gz":
        dt.to_nii_gz(final_arr, name, out_path)

if __name__ == "__main__":
    print("Path of your file.")
    while True:
        in_path = input()
        if os.path.isfile(in_path):
            break
        print("Path doesn't exists, Try again.")
    
    print("Path for output.")
    while True:
        out_path = input()
        if os.path.exists(out_path):
            break
        print("Path doesn't exists, Try again.")
    main(in_path, out_path)
    logging.info("Complete!")
