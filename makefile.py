import os
import logging
from google_drive_downloader import GoogleDriveDownloader as gdd
import configs as cfgs

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def main(with_dataset):
    if os.path.exists(cfgs.base_data_path) is False:
        logging.info("Allocating '{:}'".format(cfgs.base_data_path))
        os.makedirs(cfgs.base_data_path)

    if os.path.exists(cfgs.inpath_imgs_md) is False:
        logging.info("Allocating '{:}'".format(cfgs.inpath_imgs_md))
        os.makedirs(cfgs.inpath_imgs_md)

    if os.path.exists(cfgs.inpath_rois_md) is False:        
        logging.info("Allocating '{:}'".format(cfgs.inpath_rois_md))
        os.makedirs(cfgs.inpath_rois_md)

    if os.path.exists(cfgs.path_train_images) is False:
        logging.info("Allocating '{:}'".format(cfgs.path_train_images))
        os.makedirs(cfgs.path_train_images)

    if os.path.exists(cfgs.path_train_rois) is False:        
        logging.info("Allocating '{:}'".format(cfgs.path_train_rois))
        os.makedirs(cfgs.path_train_rois)

    if os.path.exists(cfgs.path_val_images) is False:        
        logging.info("Allocating '{:}'".format(cfgs.path_val_images))
        os.makedirs(cfgs.path_val_images)

    if os.path.exists(cfgs.path_val_rois) is False:        
        logging.info("Allocating '{:}'".format(cfgs.path_val_rois))
        os.makedirs(cfgs.path_val_rois)

    if os.path.exists(cfgs.path_testset) is False:
        logging.info("Allocating '{:}'".format(cfgs.path_testset))
        os.makedirs(cfgs.path_testset)

    if os.path.exists(cfgs.outpath_pred) is False:
        logging.info("Allocating '{:}'".format(cfgs.outpath_pred))
        os.makedirs(cfgs.outpath_pred)

    if with_dataset == "1"
        img_id = "18bNpMcGA0EOqvwJSQEzX5f_vAWtkWoz1"
        roi_id = "1plo0Hl3N86kE50gi55miiVdHS8lY1YrL"
        gdd.download_file_from_google_drive(file_id=img_id,
                                        dest_path=cfgs.inpath_imgs_md+"raw_images.zip",
                                        unzip=True)
        gdd.download_file_from_google_drive(file_id=roi_id,
                                        dest_path=cfgs.inpath_rois_md+"raw_rois.zip",
                                        unzip=True)
        os.remove(cfgs.inpath_imgs_md+"raw_images.zip")
        os.remove(cfgs.inpath_rois_md+"raw_rois.zip")

if __name__ == "__main__":
    print("Download aorta dataset? (0=No, 1=Yes)")
    main(input())
    print("Complete!")
