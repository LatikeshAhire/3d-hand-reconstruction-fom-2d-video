import os

import cv2

# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import config
import utils
from PSMNet import PSMNet
import matplotlib.pyplot as plt
import numpy as np
from dataloader.data_loader import DataLoaderSceneFlow, DataLoaderKITTI


def train(ckpt_path, train_data_loader, val_data_loader, save_model_name):
    """
    训练
    :param val_data_loader:
    :param train_data_loader:
    :param ckpt_path:
    :return:
    """
    with tf.Session() as sess:
        # Build the model
        model = PSMNet(width=config.TRAIN_CROP_WIDTH, height=config.TRAIN_CROP_HEIGHT, channels=config.IMG_N_CHANNEL,
                       head_type=config.HEAD_STACKED_HOURGLASS, batch_size=config.TRAIN_BATCH_SIZE)
        model.build_net()
        
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("total parameters: ",total_parameters)
        
        
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('log/', sess.graph)
        global_step = 0
        # train
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, save_path=ckpt_path)
        for epoch in range(1, config.TRAIN_EPOCH + 1):
            epoch_loss = 0
            step_ave_loss = 0
            for step, (imgL_crop, imgR_crop, groundtruth) in enumerate(train_data_loader.generator(is_training=True)):
                loss, log = model.train(
                    sess,
                    left_imgs=imgL_crop,
                    right_imgs=imgR_crop,
                    disp_gt=groundtruth
                )
                epoch_loss += loss
                step_ave_loss += loss
                global_step += 1
                # save log
                writer.add_summary(log, global_step)
                if step % config.LOG_INTERVAL == 0 and step > 0:
                    print('EPOCH {:04d} - {:04d}/{:04d}  LOSS: {:.4f}'.format(
                        epoch, step, train_data_loader.train_size // config.TRAIN_BATCH_SIZE,
                                     step_ave_loss / config.LOG_INTERVAL
                    ))
                    step_ave_loss = 0
            print('EPOCH LOSS: {}'.format(epoch_loss / step))

            # verify
            val(sess, model, val_data_loader, vis=False, save_fig=False)

            # save model

            saver.save(sess, save_path='./ckpt/{}.ckpt'.format(save_model_name), global_step=epoch)


def val(sess, model: PSMNet, data_loader, vis=False, save_fig=False):
    """
    verify
    :param sess:
    :param model:
    :param vis:
    :return:
    """
    error_total = []
    # verify
    for step, (imgL_crop, imgR_crop, groundtruth) in enumerate(data_loader.generator(is_training=False)):
        prediction = model.predict(
            sess,
            left_imgs=imgL_crop,
            right_imgs=imgR_crop,
        )

        if groundtruth is not None:
            # Calculation error
            error_npx = utils.compute_npx_error(prediction, groundtruth, n=3)
            error_total.append(error_npx)
            print('npx-Error: {}'.format(error_npx))

        # visualization
        for img_id in range(len(prediction)):
            if save_fig:
                cv2.imwrite('./vis/{:05d}_{:03d}.png'.format(step, img_id), prediction[img_id])

            if vis:
                plt.gcf().set_size_inches(10, 4)
                plt.subplot(1, 2, 1)
                plt.imshow(imgL_crop[img_id] * 0.25 + 0.4)
                plt.subplot(1, 2, 2)
                plt.imshow(prediction[img_id], cmap=plt.get_cmap('rainbow'))
                plt.tight_layout()
                plt.show()

    print('Total average error: {}'.format(np.average(error_total)))
    return np.average(error_total)


def finetune():
    """
    Tuning on the KITTI dataset
    :return:
    """
    train(
        ckpt_path='./ckpt/scene_flow_hgls.ckpt-7',
        train_data_loader=DataLoaderKITTI(batch_size=config.TRAIN_BATCH_SIZE, max_disp=config.MAX_DISP),
        val_data_loader=DataLoaderKITTI(batch_size=config.VAL_BATCH_SIZE, max_disp=config.MAX_DISP),
        save_model_name='KITTI_hgls'
    )


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # tests(ckpt_path='./ckpt/scene_flow_hgls.ckpt-32', vis=True)
    # train(
    #     ckpt_path='./ckpt/scene_flow_hgls.ckpt-6',

    #     train_data_loader=DataLoaderSceneFlow(batch_size=config.TRAIN_BATCH_SIZE, max_disp=config.MAX_DISP),
    #     val_data_loader=DataLoaderSceneFlow(batch_size=config.VAL_BATCH_SIZE, max_disp=config.MAX_DISP),
    #     save_model_name='scene_flow_hgls'
    # )

    finetune()
