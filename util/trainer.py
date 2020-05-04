from segmentation_models import Unet
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, LearningRateScheduler
from util.data_util import all_path, find_best_path
from keras.utils.generic_utils import get_custom_objects

import datetime
import logging
import numpy as np
import os
import shutil
import util.predict_callback as pc
import math
import configs as cfgs

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class Trainer(object):

    def __init__(self, backbone_name, train_gen, val_gen):
        self.backbone_name = backbone_name
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.restore_from_epoch = 0

    def initialize(self, restore):
        if not restore:
            logging.info("Removing '{:}'".format(cfgs.outpath_pred_each_epoch))
            shutil.rmtree(cfgs.outpath_pred_each_epoch, ignore_errors=True)
            logging.info("Removing '{:}'".format(cfgs.base_model_path))
            shutil.rmtree(cfgs.base_model_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(cfgs.outpath_tensorboard))
            shutil.rmtree(cfgs.outpath_tensorboard, ignore_errors=True)

        if not os.path.exists(cfgs.outpath_pred_each_epoch):
            logging.info("Allocating '{:}'".format(cfgs.outpath_pred_each_epoch))
            os.makedirs(cfgs.outpath_pred_each_epoch)

        if not os.path.exists(cfgs.base_model_path):
            logging.info("Allocating '{:}'".format(cfgs.base_model_path))
            os.makedirs(cfgs.base_model_path)
            os.makedirs(cfgs.best_model_path)
            os.makedirs(cfgs.last_model_path)

        if not os.path.exists(cfgs.outpath_pred):
            logging.info("Allocating '{:}'".format(cfgs.outpath_pred))
            os.makedirs(cfgs.outpath_pred)

    def train(self, optimizer, loss, metrics, epoch=10, restore=False):
        self.initialize(restore)

        if restore:
            set_custom_object(loss, metrics)
        else:
            size = "{:}*{:}".format(cfgs.train_size_ny, cfgs.train_size_nx)
            detail_writer(cfgs.base_model_path, self.backbone_name, optimizer, loss, size, cfgs.divide_stride, metrics)

        model = Unet(backbone_name= self.backbone_name, 
             input_shape=(cfgs.train_size_ny, cfgs.train_size_nx, 1), 
             freeze_encoder=False, 
             decoder_use_batchnorm=True,
             classes=cfgs.n_classes,
             activation="sigmoid",
             encoder_weights=None)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if restore:
            last_model = all_path(cfgs.last_model_path+"*")

            self.restore_from_epoch, path = find_best_path(last_model)
            if path != "":
                epoch = epoch + self.restore_from_epoch
                model.load_weights(path)

        model.summary()

        last_model = ModelCheckpoint(filepath=cfgs.last_model_path+cfgs.base_model_name, 
            monitor="val_" + cfgs.save_model_monitor, 
            verbose=1, 
            save_best_only=False, 
            save_weights_only=True, 
            mode='max')
        
        best_model = ModelCheckpoint(filepath=cfgs.best_model_path+cfgs.base_model_name, 
            monitor="val_" + cfgs.save_model_monitor, 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='max')

        csv_logger = CSVLogger(cfgs.base_model_path+'train_history.csv', append=restore, separator="/")

        pred_each_epoch = pc.Histories(cfgs, with_gt=True)

        time = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%p')

        tb = TensorBoard(log_dir=cfgs.outpath_tensorboard + "{:}".format(time))

        if cfgs.lr_schedule_mode is False:
            callbacks_list =[best_model, last_model, csv_logger, pred_each_epoch, tb]
        else:
            lr_decay = LearningRateScheduler(lr_scheduler)
            callbacks_list =[best_model, last_model, csv_logger, pred_each_epoch, tb, lr_decay]
        
        model.fit_generator(generator=self.train_gen, 
            validation_data=self.val_gen, 
            steps_per_epoch=self.train_gen.__len__(), 
            epochs=epoch, 
            initial_epoch=self.restore_from_epoch, 
            callbacks=callbacks_list,
            workers=4)
        
        return model

def lr_scheduler(epoch, lr, decay_epoch=5, max_low_decay=1e-4):
    if epoch != 0 and epoch % decay_epoch == 0:
        lr /= 2.
    if lr < max_low_decay:
        lr = max_low_decay

    print("lr:", lr)
    
    return lr

def set_custom_object(loss, metrics=None):
    cus_name = str(loss)
    cus_name = cus_name[cus_name.find(" ")+1:cus_name.find(" at ")]
    get_custom_objects().update({cus_name: loss})

    if metrics is not None:
        for cus_met in(metrics):
            cus_name = str(cus_met)
            cus_name = cus_name[cus_name.find(" ")+1:cus_name.find(" at ")]
            get_custom_objects().update({cus_name: cus_met})

def detail_writer(outpath, backbone_name, optimizer, loss, size, stride, metrics):
    op_name = str(optimizer)[str(optimizer).rfind(".")+1:str(optimizer).find(" ")]
    loss = str(loss)
    loss = loss[loss.find(" ")+1:loss.find(" at ")]

    outpath = outpath + "detail.txt"

    f = open(outpath, "w")
    f.write("Backbone: {:}\nOptimizer: {:} {:}\nLoss: {:}\nTrain-size: {:}\nStride: {:}\n===Metrics===\n".format(backbone_name, 
        op_name, 
        optimizer.get_config(), 
        loss, 
        size, 
        stride))

    for name in(metrics):
        name = str(name)
        name = name[name.find(" ")+1:name.find(" at ")]
        f.write(name+"\n")
    f.close()
