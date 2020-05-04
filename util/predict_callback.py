import keras
import numpy as np
import cv2
from util.data_util import (save_img, combine_img_prediction, combine_img_prediction_gt, 
    process_data, divide_image, find_best_piece, tail_word, cal_step, ori_position, pred_in_ori_position)

class Histories(keras.callbacks.Callback):
    def __init__(self, cfgs, with_gt):
        self.cfgs = cfgs
        self.with_gt = with_gt

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}, threshold=0.5):
        num = len(self.cfgs.check_images)

        for i in range(num):
            x = process_data(cv2.imread(self.cfgs.check_images[i], 0))

            div_x = divide_image(x)

            pred = self.model.predict(div_x.reshape(div_x.shape[0], div_x.shape[1], div_x.shape[2], 1), batch_size=self.cfgs.predict_batch_size)
            pred = process_data(pred[:,...,1])
            num_best_piece = find_best_piece(pred)
            pred = pred[num_best_piece, ]

            step_h, step_w = cal_step(x)
            t, b, l, r = ori_position(x, num_best_piece)
            pred_b, pred_r = pred_in_ori_position(x, b, r)

            #tmp_pred = np.copy(x[i, ])
            tmp_pred = np.zeros([x.shape[0], x.shape[1]], dtype=np.float32)
            tmp_pred[t:b, l:r] = pred[:pred_b, :pred_r]

            if self.with_gt:
                y = process_data(cv2.imread(self.cfgs.check_rois[i], 0))

                img = combine_img_prediction_gt(x , y, tmp_pred)
            else:
                img = combine_img_prediction(x , tmp_pred)

            name = tail_word(self.cfgs.check_images[i])
            name = name[:name.rfind(".")]
            save_img(img, self.cfgs.outpath_pred_each_epoch, "epoch_{:03}-".format(epoch) + name)

        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
        