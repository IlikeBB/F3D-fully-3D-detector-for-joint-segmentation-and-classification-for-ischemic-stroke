import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from skfuzzy import cmeans
import nibabel as nib
from skimage import morphology
import tensorflow as tf
import random
def rotate(volume):
    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-10, -5, 5, 10]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def horizontal_flip(volume):
    def horizontal_(volume):
        rate = 0.5
        if np.random.rand() < rate:
            volume = volume[:,::-1,:]
        return volume
    augmented_volume = tf.numpy_function(horizontal_, [volume], tf.float32)
    return augmented_volume

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = horizontal_flip(volume)
#     volume = vertical_flip(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def train_preprocessing2(volume, label, weight):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = horizontal_flip(volume)
#     volume = vertical_flip(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label, weight


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label



def standardization(image):
    # Get brain mask
    mask = image == 0
    
    selem = morphology.disk(2)
    
    segmentation = morphology.dilation(mask, selem)
    labels, label_nb = ndimage.label(segmentation)

    mask = labels == 0
    
    mask = morphology.dilation(mask, selem)
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, selem)
    
    # Normalize
    
    mean = image[mask].mean()
    std = image[mask].std()
    
    normalized_image = (image - mean) / std
    
    return normalized_image

def fcm_class_mask(image, brain_mask):
    # We want 3 classes with a maximum of 50 iterations
    [class_centers, partitioned_matrix, _, _, _, _, _] = cmeans(image[brain_mask].reshape(-1, len(brain_mask[brain_mask])),
                                              3, 2, 0.005, 50)
    matrix_list = [partitioned_matrix[class_num] for class_num, _ in sorted(enumerate(class_centers),
                                                                          key=lambda x: x[1])] 
    mask = np.zeros(image.shape + (3,))
    for index in range(3):
        mask[..., index][brain_mask] = matrix_list[index]
    
    return mask

def get_white_matter_mask(image, threshold=0.8):
    # Get brain mask
    mask = image == 0
    selem = morphology.disk(2)
    segmentation = morphology.dilation(mask, selem)
    labels, label_nb = ndimage.label(segmentation)

    mask = labels == 0
    mask = morphology.dilation(mask, selem)
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, selem)
    
    # Get White Matter mask
    matrix_mask = fcm_class_mask(image, mask)
    white_matter_mask = matrix_mask[..., 2] > threshold
    
    return white_matter_mask 

def fcm_normalization(image, norm_value=1):
    white_matter_mask = get_white_matter_mask(image)
    wm_mean = image[white_matter_mask == 1].mean()
    
    normalized_image = (image / wm_mean) * norm_value
    
    return normalized_image

def remove_noise_from_image(file_path):
    image = nib.load(file_path)
    if len(image.shape) == 4:
        image = image.get_fdata()
        width,height,queue,_ = image.shape
        image = image[:,:,:,1]
        image = np.reshape(image,(width,height,queue))
    else:
        image = image.get_fdata()
        pass
    image2 = image.copy()
    shape = image.shape
    for i in range(shape[2]):
        image_2d = image[:, :, i]
        mask = image_2d <=10

        selem = morphology.disk(2)

        segmentation = morphology.dilation(mask, selem)
        labels, label_nb = ndimage.label(segmentation)

        mask = labels ==0
        mask = morphology.dilation(mask, selem)
        mask = ndimage.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, selem)

        clean_image = mask * image_2d
        image[:, :, i] = standardization(clean_image)
    print(file_path)
    print(image.shape)
    return image

def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
#     data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

def resize_volume(img,size,depth):

    """Resize across z-axis"""
    # Set the desired depth
#     desired_depth = 64
#     desired_width = 128
#     desired_height = 128
#     # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Rotate
    img = ndimage.rotate(img, 180, reshape=False, mode="nearest")
    # Resize across z-axis
    img = ndimage.zoom(img, (size/current_height, size/current_width, 1), mode="nearest")
#     print(img.shape)
    return img

# def normalize(volume):
#     """Normalize the volume"""
#     min = -1024
#     max = 2000
# #     volume[volume < min] = min
# #     volume[volume > max] = max
# #     volume = (volume - min) / (max - min)
#     volume = volume.astype("float32")
#     return volume

    
# def resize_volume(img,size,depth):
#     """Resize across z-axis"""
#     # Set the desired depth
# #     desired_depth = 64
# #     desired_width = 128
# #     desired_height = 128
#     desired_depth = depth
#     desired_width = size
#     desired_height = size
# #     # Get current depth
#     current_depth = img.shape[-1]
#     current_width = img.shape[0]
#     current_height = img.shape[1]
# #     # Compute depth factor
#     depth = current_depth / desired_depth
#     width = current_width / desired_width
#     height = current_height / desired_height
#     depth_factor = 1 / depth
#     width_factor = 1 / width
#     height_factor = 1 / height
#     # Rotate
#     img = ndimage.rotate(img, 180, reshape=False)
#     # Resize across z-axis
#     img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
# #     print(img.shape)
#     return img