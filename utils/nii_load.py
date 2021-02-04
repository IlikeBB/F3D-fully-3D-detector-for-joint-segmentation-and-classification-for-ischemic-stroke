import nibabel as nib
from skimage import morphology
from scipy import ndimage
from PIL import Image
import numpy as np

size = 192
depth = 32
def normalize(volume, norm_type):
    if norm_type == 'zero_mean':
        img_o = np.float32(volume.copy())
        m = np.mean(img_o)
        s = np.std(img_o)
        volume = np.divide((img_o - m), s)
    elif norm_type == 'div_by_max':
        volume = np.divide(volume, np.percentile(volume,98))
        
    elif norm_type == 'onezero':
        for channel in range(volume.shape[-1]):
            volume_temp = volume[..., channel]
            volume_temp = (volume_temp - np.min(volume_temp)) / (np.max(volume_temp)-np.min(volume_temp))

            volume[..., channel] = volume_temp
    volume = volume.astype("float32")
    return volume

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
    shape = image.shape
    for i in range(shape[2]):
        image_2d = image[:, :, i]
#         mask = image_2d <=20
        mask = image_2d<=10
        selem = morphology.disk(2)

        segmentation = morphology.dilation(mask, selem)
        labels, label_nb = ndimage.label(segmentation)

        mask = labels ==0
        mask = morphology.dilation(mask, selem)
        mask = ndimage.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, selem)

        image[:, :, i] = mask * image_2d
    image = normalize(image,"onezero")
    return image

def resize_volume(img,size,depth):
    """Resize across z-axis"""
    # Set the desired depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
        # Rotate img shape = (height, wight, depth)
    for i in range(img.shape[2]):
        img[:,:,i] = np.fliplr(np.flipud(img[:,:,i]))
#     img = ndimage.rotate(img, 180, reshape=False, mode="nearest")
    img = ndimage.zoom(img, (size/current_height, size/current_width, 1), order=0)
    return img

def process_scan(path):
#     # Resize width, height and depth
    volume = remove_noise_from_image(path)
    volume = resize_volume(volume,size,depth)
#   add only black background mri image
    if volume.shape[2]!=depth:
        add_black_num = depth - volume.shape[2]
        volume = np.transpose(volume)
        for i in range(add_black_num):
            add_black_ = np.expand_dims(np.zeros((volume.shape[2],volume.shape[2])),axis=0)
            volume = np.concatenate((volume, add_black_), axis = 0)
        volume = np.transpose(volume)
    volume = np.transpose(volume)
#     print(path)
#     print(f"rebuild shape: {volume.shape}")
    return volume
def mask_scan(path):
#     print(path)
    image = nib.load(path)
    
    if len(image.shape) == 4:
        image = image.get_fdata()
        width,height,queue,_ = image.shape
        image = image[:,:,:,1]
        image = np.reshape(image,(width,height,queue))
    else:
        image = image.get_fdata()
        pass
    image = resize_volume(image,size,depth)
    shape = image.shape
#   add only black background mri image
    if image.shape[2]!=depth:
        add_black_num = depth - image.shape[2]
        image = np.transpose(image)
        for i in range(add_black_num):
            add_black_ = np.expand_dims(np.zeros((image.shape[2],image.shape[2])),axis=0)
            image = np.concatenate((image, add_black_), axis = 0)
        image = np.transpose(image)
    image = np.transpose(image)
#     print(path)
#     print(f"rebuild shape: {image.shape}")
    return image