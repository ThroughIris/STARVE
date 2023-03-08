class LossParam:
    """
    Hyper-parameters for utils/losses.py
    """
    content_weight = 100  # alpha
    style_weight = 1  # beta
    tv_weight = 8.5E-5  # total variation loss weight

    temporal_weight = 2e1  # gamma
    J = [1, 10, 20, 40]  # long-term consistency chosen frame

    use_temporal_pass = 2  # from which pass to use short-term temporal loss
    blend_factor = 0.5  # delta

    print_loss = False  # when False, will run 1.5~2x faster
