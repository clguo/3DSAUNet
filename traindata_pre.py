import numpy as np
import nibabel as nib
import cv2
import os
import glob
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fftn, ifftn

path1 = 'WMHdataset/training/Utrecht/'
path2 = 'WMHdataset/training/Amsterdam/GE3T/'
path3 = 'WMHdataset/training/Singapore/'

def elasticDeform(imageVolume, maskVolume, sigma, alpha, interMethod='linear', extraMethod='nearest'):
    # Elastic deformation of 3D images using random displacement fields

    # imageVolume: 3D image volume
    # maskVolume: 3D mask volume
    # sigma: elasticity coefficient for smoothing the random field
    # alpha: scaling factor controlling the intensity of the deformation
    # interMethod: interpolation method such as 'linear', 'cubic', 'spline', 'nearest'
    # extraMethod: extrapolation method such as 'linear', 'cubic', 'spline', 'nearest'

    # Generating a random displacement field
    imageHeight, imageWidth, imageDepth = imageVolume.shape
    dx = 2 * np.random.rand(imageHeight, imageWidth, imageDepth) - 1
    dy = 2 * np.random.rand(imageHeight, imageWidth, imageDepth) - 1
    dz = 2 * np.random.rand(imageHeight, imageWidth, imageDepth) - 1

    # Smoothing and scaling the field
    kernelSize = 2 * np.ceil(2 * sigma) + 1
    dx = alpha * gaussian_filter(dx, sigma, mode='constant', cval=0)
    dy = alpha * gaussian_filter(dy, sigma, mode='constant', cval=0)
    dz = alpha * gaussian_filter(dz, sigma, mode='constant', cval=0)

    # Applying the random displacement (elastic distortion)
    x, y, z = np.ogrid[:imageHeight, :imageWidth, :imageDepth]

    # Elastic deformation for image volume
    F_image = RegularGridInterpolator((x[:, 0, 0], y[0, :, 0], z[0, 0, :]), imageVolume, method=interMethod, bounds_error=False, fill_value=None)
    queryPoints_image = np.array([x + dx, y + dy, z + dz]).transpose((1, 2, 3, 0))
    imageVolume_deformed = F_image(queryPoints_image)

    # Elastic deformation for mask volume
    F_mask = RegularGridInterpolator((x[:, 0, 0], y[0, :, 0], z[0, 0, :]), maskVolume, method='nearest', bounds_error=False, fill_value=None)
    queryPoints_mask = np.array([x + dx, y + dy, z + dz]).transpose((1, 2, 3, 0))
    maskVolume_deformed = F_mask(queryPoints_mask)

    return imageVolume_deformed, maskVolume_deformed

def biasField(imageVolume):
    # Random intensity inhomogeneity generation in 3D images

    imageHeight, imageWidth, imageDepth = imageVolume.shape
    X, Y, Z = np.meshgrid(np.arange(1, imageHeight + 1), np.arange(1, imageWidth + 1), np.arange(1, imageDepth + 1), indexing='ij')
    x0 = np.random.randint(1, imageHeight + 1)
    y0 = np.random.randint(1, imageWidth + 1)
    z0 = np.random.randint(1, imageDepth + 1)
    G = 1 - ((X - x0) ** 2 / (imageHeight ** 2) + (Y - y0) ** 2 / (imageWidth ** 2) + (Z - z0) ** 2 / (imageDepth ** 2))

    imageVolume = G * imageVolume.astype(float)

    return imageVolume.astype(imageVolume.dtype)

def resize_image_and_mask(image, mask, target_size):
    # Get the original size of the image and mask
    original_size = np.array(image.shape)
    # Calculate size difference
    size_diff = target_size - original_size

    # Determine whether to perform padding or cropping
    if np.any(size_diff < 0):
        # Crop image and mask by removing the excess parts
        start = -size_diff // 2
        end = start + target_size
        resized_image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        resized_mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    else:
        # Create new image and mask with padding
        pad_before = size_diff // 2
        pad_after = size_diff - pad_before
        pad_width = [(pad_before[i], pad_after[i]) for i in range(len(size_diff))]
        resized_image = np.pad(image, pad_width, mode='constant')
        resized_mask = np.pad(mask, pad_width, mode='constant')

    return resized_image, resized_mask

def restore_image_and_mask(resized_image, resized_mask, original_size):
    # Calculate size difference
    size_diff = original_size - np.array(resized_image.shape)
    # Calculate crop boundaries
    start = -size_diff // 2
    end = start + original_size
    # Restore image and mask to original size
    restored_image = resized_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    restored_mask = resized_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    return restored_image, restored_mask

def motion_ghosting(imageVolume, alpha, numReps, p):
    imageHeight, imageWidth, imageDepth = imageVolume.shape

    imageVolume = fftn(imageVolume, (imageHeight, imageWidth, imageDepth))

    if p == 1:  # along x-axis
        imageVolume[::numReps, :, :] = alpha * imageVolume[::numReps, :, :]
    elif p == 2:  # along y-axis
        imageVolume[:, ::numReps, :] = alpha * imageVolume[:, ::numReps, :]
    elif p == 3:  # along z-axis
        imageVolume[:, :, ::numReps] = alpha * imageVolume[:, :, ::numReps]

    imageVolume = np.abs(ifftn(imageVolume, (imageHeight, imageWidth, imageDepth)))

    return imageVolume.astype(imageVolume.dtype)

img_row = 256
img_col = 256

def image_processing(file_path):
    data_path = sorted(os.listdir(file_path), key=lambda x: int(os.path.splitext(x)[0]))
    flair_dataset = []
    mask_dataset = []
    for i in data_path:
        img_path = os.path.join(file_path, i, 'pre')
        mask_path = os.path.join(file_path, i)

        for image, mask in zip(glob.glob(img_path + '/FLAIR*'), glob.glob(mask_path + '/wmh*')):
            flair_img = nib.load(image)
            flair_data = flair_img.get_fdata()
            mask_img = nib.load(mask)
            mask_data = mask_img.get_fdata()

            flair_resized, mask_resized = resize_image_and_mask(flair_data, mask_data, (256, 256, 128))
            flair_dataset.append(flair_resized)
            mask_dataset.append(mask_resized)

            # Rotation
            M1 = cv2.getRotationMatrix2D((img_row / 2, img_col / 2), 90, 1)
            flair_rotate = cv2.warpAffine(flair_resized, M1, (img_row, img_col))
            flair_dataset.append(flair_rotate)
            mask_rotate = cv2.warpAffine(mask_resized, M1, (img_row, img_col), flags=cv2.INTER_NEAREST)
            mask_dataset.append(mask_rotate)

            # Elastic deformation
            sigma = (20 - 10) * np.random.rand() + 10
            alpha = (200 - 100) * np.random.rand() + 100
            X1, Y1 = elasticDeform(flair_resized, mask_resized, sigma=sigma, alpha=alpha)
            flair_dataset.append(X1)
            mask_dataset.append(Y1)

            # Intensity inhomogeneity
            X2 = biasField(flair_resized)
            flair_dataset.append(X2)
            mask_dataset.append(mask_resized)

            # Motion ghosting
            alpha = (0.7 - 0.5) * np.random.rand() + 0.5
            numReps = np.random.randint(2, 4)
            dim = 2
            X3 = motion_ghosting(imageVolume=flair_resized, alpha=alpha, numReps=numReps, p=dim)
            flair_dataset.append(X3)
            mask_dataset.append(mask_resized)

            # Gaussian noise
            X4 = flair_resized + np.random.normal(0, 0.1, size=flair_resized.shape)
            flair_dataset.append(X4)
            mask_dataset.append(mask_resized)

    flair_array = np.array(flair_dataset)
    mask_array = np.array(mask_dataset)
    return flair_array, mask_array

utrecht_flair, utrecht_mask = image_processing(path1)
amsterdam_flair, amsterdam_mask = image_processing(path2)
singapore_flair, singapore_mask = image_processing(path3)

np.save('train/ori/utrecht_flair(256)aug.npy', utrecht_flair)
np.save('train/ori/utrecht_mask(256)aug.npy', utrecht_mask)
np.save('train/ori/amsterdam_flair(256)aug.npy', amsterdam_flair)
np.save('train/ori/amsterdam_mask(256)aug.npy', amsterdam_mask)
np.save('train/ori/singapore_flair(256)aug.npy', singapore_flair)
np.save('train/ori/singapore_mask(256)aug.npy', singapore_mask)

print("finish")
