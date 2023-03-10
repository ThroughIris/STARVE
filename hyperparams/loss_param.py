class LossParam:
    """
    Hyper-parameters for utils/losses.py
    """
    content_weight = 5  # alpha
    style_weight = 10000  # beta
    tv_weight = 0.001  # total variation loss weight

    temporal_weight = 0.001  # gamma
    J = [1, 10, 20, 40]  # long-term consistency chosen frame

    use_temporal_pass = 8  # from which pass to use short-term temporal loss
    blend_factor = 0.5  # delta

    print_loss = False  # when False, will run 1.5~2x faster
