# Import the necessary libraries
import os
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
from utils.img_utils import load_img, preprocess_img, deprocess_img, tensor_to_image
from utils.losses import compute_loss, gram_matrix, style_loss, content_loss, tv_loss, temporal_loss, compute_long_term_loss
from models import get_model
from utils.optflow import compute_opt_flow
from utils.util import create_video_from_frames, save_frames_to_folder

# Define the command line arguments
parser = argparse.ArgumentParser(description='STyLE-Net Adaptive Refinement')
parser.add_argument('--content', help='Path to content image', required=True)
parser.add_argument('--style', help='Path to style image', required=True)
parser.add_argument('--output_dir', help='Output directory', required=True)
parser.add_argument('--iterations', help='Number of iterations', default=1000, type=int)
parser.add_argument('--width', help='Image width', default=512, type=int)
parser.add_argument('--save_every', help='Save image every N iterations', default=100, type=int)
parser.add_argument('--gpu', help='GPU index to use', default=0, type=int)
parser.add_argument('--use_opt_flow', help='Use optical flow for temporal loss', action='store_true')
parser.add_argument('--opt_flow_dir', help='Directory to store optical flow data')
parser.add_argument('--resume', help='Resume from previous checkpoint', action='store_true')

# Define the loss parameters
class LossParam:
    """
    Hyper-parameters for utils/losses.py
    """
    content_weight = 5e0  # alpha
    style_weight = 1e4  # beta
    tv_weight = 1e-3  # total variation loss weight

    temporal_weight = 2e2  # gamma
    J = [1, 2, 5]  # long-term consistency chosen frame

    use_temporal_pass = 2  # from which pass to use short-term temporal loss
    blend_factor = 0.5  # delta

    print_loss = False  # when False, will run 1.5~2x faster

# Parse the command line arguments
args = parser.parse_args()

# Set the GPU index
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Load the content and style images
content_image = load_img(args.content, target_size=(args.width, args.width))
style_image = load_img(args.style, target_size=(args.width, args.width))

# Preprocess the images
content_image = preprocess_img(content_image)
style_image = preprocess_img(style_image)

# Define the model
model = get_model()

# Initialize the optimizer
optimizer = tf.optimizers.Adam(learning_rate=5e-2, beta_1=0.99, epsilon=1e-1)

# Define the checkpoint directory
now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
checkpoint_dir = os.path.join(args.output_dir, 'checkpoints', now)

# Define the checkpoint prefix
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

# Define the file paths for saving the stylized images and video
output_dir = os.path.join(args.output_dir, 'output', now)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image_prefix = os.path.join(output_dir, 'output')
video_path = os.path.join(args