import numpy as np
from skimage.metrics import structural_similarity


def psnr(fake_img, real_img, obj_mask_1d=None):
    """
    Peak Signal-To-Noise-Ratio

    Images have dimension HxWx3 and are in [0,1]
    
    Here: MAX := 1.0
    PSNR = 10 * log10(MAX^2 / MSE) = 20 * log10(MAX / sqrt(MSE))
    """
    assert fake_img.shape == real_img.shape
    if obj_mask_1d is not None:
        fake_img = fake_img.reshape(-1, 3)[obj_mask_1d]
        real_img = real_img.reshape(-1, 3)[obj_mask_1d]
    mse = np.mean((fake_img - real_img)**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def epoch_psnr(epoch_mse):
    return -10 * np.log10(epoch_mse)


def dssim(fake_image, real_image):
    """
    Structural Dissimilarity based on Structural Similarity Index Metric (SSIM)
    """
    assert fake_image.shape == real_image.shape and fake_image.shape[2] == 3
    return (1 - structural_similarity(fake_image, real_image, multichannel=True)) / 2
