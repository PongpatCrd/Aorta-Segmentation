""" setting """
# General
base_data_path = "./dataset/"
path_train_images = base_data_path + "train_images/"
path_train_rois = base_data_path + "train_rois/"
path_val_images = base_data_path + "val_images/"
path_val_rois = base_data_path + "val_rois/"
path_testset = base_data_path + "__test__/"
lr_schedule_mode = True

# metrics
loss = "tversky_loss"
metrics = ["hard_dice_coef_th", "precision_th", "recall_th", "f1_th", "accuracy"]

#input-output setting
output_type = ".nii.gz"	# .nii.gz or .npz
train_size_ny = 256     
train_size_nx = 256     
divide_stride = 64		#can not be zero
n_classes = 2
threshold = 0.5

# make_dataset.py
inpath_imgs_md = base_data_path + "raw_images/"
inpath_rois_md = base_data_path + "raw_rois/"
size_ny_md = 512    #0 = same to original
size_nx_md = 512    #0 = same to original

# trainer.py
train_batch_size = 4
base_model_path = "./t_model/"
best_model_path = base_model_path + "best/"
last_model_path = base_model_path + "last/"
check_images = []
check_rois = []
outpath_tensorboard = "./logsTB/"

save_model_monitor = "hard_dice_coef_th"
base_model_name = "epoch_{epoch:03}-hard_dice_coef_th_{val_hard_dice_coef_th:.4f}.hdf5"

# predict_util.py
predict_batch_size = 8
outpath_pred = "./prediction/"
outpath_pred_each_epoch = "./prediction_each_epoch/"