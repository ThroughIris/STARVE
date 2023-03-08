class DatasetParam:
    """
    Hyper-parameters for utils/dataset.py
    """
    img_w = 720  # image width, should be EVEN, or the final .mp4 video may be crashed
    img_h = 1280  # image height, should be EVEN, or the final .mp4 video may be crashed

    use_video = True  # Style transfer with a video or an image. True: video; False: image
    content_img_path = r'demo/CaseStudy02_alone_v018_Still.jpg'
    video_path = r'demo/CaseStudy02_alone_v018_ShortTest.mp4'
    style_img_path = r'demo/GraphicDesign_14.jpg'

    img_fmt = 'jpg'  # frame image format
    video_fps = 30  # select `video_fps` frames per second

    # method to calculate optic flow
    # 'dm_df2': DeepMatching + DeepFlow2, the best quality
    # 'df2': DeepFlow2 only, 2~3s per frame
    # 'std': Sparse to Dense, 10 fps
    # 'liteF': liteFlowNet
    optic_flow_method = 'std'

    # method to initialize the stylized image
    # useful when not using multi-pass,
    # or in the first pass when using multi-pass
    # 'image': use the current video frame image
    # 'image_flow_warp': warp the previous stylized image with optic flow
    # 'random': normal distribution
    init_generated_image_method = 'image'
