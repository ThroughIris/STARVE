from models.model import STARVE
from utils.dataset import load_img, tensor_to_image, \
    video_to_frames, frames_to_video, make_optic_flow, init_generated_image, \
    make_warped_images_for_temporal_loss, make_consistency_for_temporal_loss
from utils.optimizers import get_optimizer
from utils.losses import style_content_loss, tv_loss, temporal_loss
from hyperparams.train_param import TrainParam
from hyperparams.dataset_param import DatasetParam
from hyperparams.loss_param import LossParam

import tensorflow as tf
import cv2
from tqdm import tqdm
from os import makedirs
from os.path import join, isdir, basename, splitext
import glob


def preparation():
    """
    Preparations before training begins.
    Including creating directories, convert a video into frames,
    calculate optic flows, etc.
    :return:
        None
    """
    if not isdir(TrainParam.output_dir):
        makedirs(TrainParam.output_dir)
    if not isdir(TrainParam.iter_img_dir):
        makedirs(TrainParam.iter_img_dir)
    if not isdir(TrainParam.stylized_img_dir):
        makedirs(TrainParam.stylized_img_dir)

    if DatasetParam.use_video:
        # convert video to frames
        video_to_frames(DatasetParam.video_path, TrainParam.video_frames_dir)
        if TrainParam.use_optic_flow:
            optic_flow_path = join(TrainParam.video_optic_flow_dir, '*.flo')
            existing_flow_files = glob.glob(optic_flow_path)
            if not existing_flow_files:
                make_optic_flow(TrainParam.video_frames_dir, TrainParam.video_optic_flow_dir)

    return


def train():
    model = STARVE()

    # get style target
    style_img_path = DatasetParam.style_img_path
    style_target = model(tf.constant(load_img(style_img_path)))['style']

    # get content image path list
    if DatasetParam.use_video:
        content_img_list = glob.glob(join(TrainParam.video_frames_dir,
                                          '*.{}'.format(DatasetParam.img_fmt)))
        content_img_list.sort(key=lambda x: int(splitext(basename(x))[0]))
    else:
        content_img_list = [DatasetParam.content_img_path]

    for n_pass in range(1, TrainParam.n_passes + 1):
        for n_img, content_img_path in enumerate(content_img_list):
            # Call tf.function each time, or there will be
            # ValueError: tf.function-decorated function tried to create variables on non-first call
            # because of issues with lazy execution.
            # https://www.machinelearningplus.com/deep-learning/how-use-tf-function-to-speed-up-python-code-tensorflow/
            if LossParam.print_loss:
                tf_train_step = train_step
            else:
                tf_train_step = tf.function(train_step)

            optimizer = get_optimizer()
            frame_idx = int(splitext(basename(content_img_path))[0]) \
                if DatasetParam.use_video else splitext(basename(content_img_path))[0]
            content_target = model(tf.constant(load_img(content_img_path)))['content']
            generated_image = init_generated_image(frame_idx, n_pass, n_img == 0) \
                if DatasetParam.use_video else tf.Variable(load_img(content_img_path))

            # component for temporal loss
            warped_images, consistency_weights = None, None
            if DatasetParam.use_video and TrainParam.use_optic_flow:
                warped_images = make_warped_images_for_temporal_loss(frame_idx, n_pass, n_img == 0)
                consistency_weights = make_consistency_for_temporal_loss(frame_idx, n_pass, n_img == 0)

            pbar = tqdm(range(TrainParam.n_step))
            pbar.set_description_str('[pass {}/{} | frame {}/{} {}]'
                                     .format(n_pass, TrainParam.n_passes,
                                             n_img + 1, len(content_img_list), basename(content_img_path)))
            for step in pbar:
                loss_dict = tf_train_step(model, generated_image, optimizer, content_target, style_target,
                                          warped_images=warped_images, consistency_weights=consistency_weights)
                los_strs_list = ["{}: {:2f}".format(k, v.item()) for k, v in loss_dict.items()]
                pbar.set_postfix_str(' | '.join(los_strs_list))
                if (step + 1) % TrainParam.draw_step == 0:
                    # save intermediate result
                    cv2.imwrite(join(TrainParam.iter_img_dir, "{}_p{}.{}"
                                     .format(step + 1, n_pass, DatasetParam.img_fmt)),
                                tensor_to_image(generated_image))
            else:
                save_path = join(TrainParam.stylized_img_dir,
                                 "{}_p{}.{}".format(frame_idx, n_pass, DatasetParam.img_fmt))
                cv2.imwrite(save_path, tensor_to_image(generated_image))

        content_img_list = content_img_list[::-1]  # reverse pass direction

    return


def train_step(model, generated_image, optimizer, content_target, style_target, **kwargs):
    """
    Each training step.
    :param model: VGG
    :param generated_image: the image that needs to update
    :param optimizer: optimizer
    :param content_target: intermediate layer outputs of the content image
    :param style_target: intermediate layer outputs of the style image
    :param kwargs:
        If use temporal loss, i.e, (DatasetParam.use_video and TrainParam.use_optic_flow) is True,
        then kwargs['warped_images'], kwargs['consistency_weights'] will be used.
    :return:
        loss_dict: Dict of loss values.
    """
    loss_dict = {}
    with tf.GradientTape() as tape:
        outputs = model(generated_image)
        loss, loss_dict_sc = style_content_loss(outputs,
                                                style_targets=style_target,
                                                content_targets=content_target)
        loss += tv_loss(generated_image)
        if LossParam.print_loss:
            loss_dict.update(loss_dict_sc)
            loss_dict['tv'] = 0
            loss_dict['tv_w'] = loss.numpy() - loss_dict['style_w'] - loss_dict['content_w']
            loss_dict['tv'] = loss_dict['tv_w'] / LossParam.tv_weight
        if DatasetParam.use_video and TrainParam.use_optic_flow:
            if kwargs['warped_images'] is not None:
                loss += temporal_loss(generated_image, kwargs['warped_images'], kwargs['consistency_weights'])
                if LossParam.print_loss:
                    loss_dict['temporal'] = 0
                    loss_dict['temporal_w'] = (loss.numpy() - loss_dict['style_w'] -
                                               loss_dict['content_w'] - loss_dict['tv_w'])
                    loss_dict['temporal'] = loss_dict['temporal_w'] / LossParam.temporal_weight
        if LossParam.print_loss:
            loss_dict['loss_w'] = loss.numpy()
    grad = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=-150, clip_value_max=150))

    return loss_dict


def post_process():
    """
    Convert the stylized frames of the final pass to a video.
    :return:
        None
    """
    if DatasetParam.use_video:
        # convert frames to videos
        frames_to_video(TrainParam.stylized_img_dir,
                        join(TrainParam.output_dir, 'stylized_{}'.format(basename(DatasetParam.video_path))))

    return


if __name__ == '__main__':
    preparation()
    train()
    post_process()
