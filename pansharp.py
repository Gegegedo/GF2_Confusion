import gdal
import scipy.ndimage as ndimage
from scipy.optimize import leastsq
import numpy as np
import cv2
"""
define block size 
"""
global ms_block_size
ms_block_size = 1024
global pan_block_size
pan_block_size = 4096
global factor
factor = 4

def compute_scale(pan_length,ms_length):
    if pan_length/ms_length<factor:
        pan_length-=pan_length%4
        ms_length=pan_length/4
    elif pan_length/ms_length>factor:
        pan_length=ms_length*4
    return int(pan_length),int(ms_length)
def resgistration(MS_path,PAN_path):
    MS_dataset = gdal.Open(MS_path)
    PAN_dataset = gdal.Open(PAN_path)
    MS_origin = MS_dataset.ReadAsArray(0, 0, MS_dataset.RasterXSize, MS_dataset.RasterYSize)
    PAN_origin = PAN_dataset.ReadAsArray(0, 0, PAN_dataset.RasterXSize, PAN_dataset.RasterYSize)
    MS_normalize = MS_origin / MS_origin.max() * 255
    MS_normalize=None

    PAN_normalize = (PAN_origin / PAN_origin.max() * 255).astype(np.uint8)
    MS_oneband = np.mean(MS_normalize, axis=0).astype(np.uint8)

    orb=cv2.ORB_create(nfeatures=5000)
    kp_MS,des_MS=orb.detectAndCompute(MS_oneband,None)
    kp_PAN,des_PAN=orb.detectAndCompute(PAN_normalize,None)
    FLANN_INDEX_LSH=6
    index_params=dict(algorithm=FLANN_INDEX_LSH,table_number=6,key_size=12,multi_probe_level=1)
    search_params=dict(check=100)
    matcher=cv2.FlannBasedMatcher(index_params,search_params)
    matchs=matcher.match(des_MS,des_PAN)
    minDis=9999.0
    for m in matchs:
        if m.distance<minDis:
            minDis=m.distance
    good_matchs=[m for m in matchs if m.distance<(minDis*5)]
    MS_cor=[kp_MS[m.queryIdx].pt for m in good_matchs]
    PAN_cor=[kp_PAN[m.trainIdx].pt for m in good_matchs]
    AffineMatrix,mask=cv2.findHomography(np.array(MS_cor),np.array(PAN_cor),cv2.RANSAC)
    return (cv2.warpPerspective(MS_origin.transpose(1,2,0),AffineMatrix,(MS_oneband.shape[1],MS_oneband.shape[0]))).tanspose(2,0,1),PAN_origin


def read_tiff(MS_path,PAN_path):
    MS_dataset=gdal.Open(MS_path)
    PAN_dataset=gdal.Open(PAN_path)
    MS=MS_dataset.ReadAsArray(0,0,MS_dataset.RasterXSize,MS_dataset.RasterYSize)
    PAN=PAN_dataset.ReadAsArray(0,0,PAN_dataset.RasterXSize,PAN_dataset.RasterYSize)
    MS_normalize=MS/MS.max()*255
    PAN_normalize=(PAN/PAN.max()*255).astype(np.uint8)
    fusion_MS = np.mean(MS_normalize, axis=0).astype(np.uint8)
    return fusion_MS,PAN_normalize,MS
    # """
    # load the remote sensing images
    # :param path: the path of images
    # :return: image, the width of image, the height of image and the number of bands
    # """
    # image = gdal.Open(path)
    # image_width = image.RasterXSize
    # image_height = image.RasterYSize
    # band_num = image.RasterCount
    #
    # return image, image_width, image_height, band_num

# define some functions used in fusion methods
def unsample(image, fac):
    """
    this method is used to up-sample the original ms image which makes ms's size equal to the size of pan image
    :param image: the original ms image
    :param fac:  the ratio between ms and pan
    :return: the up-sampled ms image
    """
    unsample_image = np.array(ndimage.zoom(image, (1, fac, fac), order=1))
    image=None

    return unsample_image


def fusion_method(ms, pan, method):
    """
    this method is used to fusion
    :param ms: the up-sampled ms image
    :param pan: the original pan image
    :param method: the specific fusion method
    :return: the fusion image
    """
    return method(ms, pan)


def fun(para, x1, x2, x3, x4):
    w1, w2, w3, w4 = para
    return w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4


def error(para, x1, x2, x3, x4, y):
    return y - fun(para, x1, x2, x3, x4)


# define the pansharp fusion method
def pan_sharp(ms, pan):
    ms_b = ms[0].flatten()
    ms_g = ms[1].flatten()
    ms_r = ms[2].flatten()
    ms_nir = ms[3].flatten()
    y = pan.flatten()
    para_initial = [0, 0, 0, 0]
    fusion_image = np.zeros((4, pan.shape[0], pan.shape[1]))
    para = leastsq(error, para_initial, args=(ms_b, ms_g, ms_r, ms_nir, y))
    w1, w2, w3, w4 = para[0]
    pan_syn = w1 * ms[0] + w2 * ms[1] + w3 * ms[2] + w4 * ms[3]
    ratio = pan / (pan_syn+0.000000000001)
    for i in range(4):
        fusion_image[i] = ms[i] * ratio

    return fusion_image


# define function which is used to write tif images
def write_tiff(im_data, im_width, im_height, im_bands, path):
    if ('uint8' or 'int8') in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif ('int16' or 'uint16') in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    data_set = driver.Create(path, im_width, im_height, im_bands, datatype)
    try:
        for i in range(im_bands):
            data_set.GetRasterBand(i + 1).WriteArray(im_data[i])
    except Exception as e:
        print(e)


def fusion_main():
    """
    the main function to fuse MS and PAN images
    :return: the fused image
    """
    # read ms and pan images
    MS,PAN=resgistration()
    p_width,p_height=PAN.shape[1],PAN.shape[0]
    m_width,m_height=MS.shape[1],MS.shape[0]
    band_num=MS.shape[0]
    # MS, m_width, m_height, band_num = read_tiff(ms_path)
    # PAN, p_width, p_height, _ = read_tiff(pan_path)
    # p_width,m_width=compute_scale(p_width,m_width)
    # p_height,m_height=compute_scale(p_height,m_height)
    # obtain the number of blocks
    pan_nx_num = (p_width - 1) // pan_block_size + 1
    pan_ny_num = (p_height - 1) // pan_block_size + 1
    # define a new array as the new fusion image
    fusion_image = np.zeros(shape=(band_num, p_height, p_width),dtype=np.uint16)

    # obtain each block ms and pan image
    for y_number in range(pan_ny_num):
        for x_number in range(pan_nx_num):
            print(y_number,x_number)
            pre_ms_x_size = 1024
            pre_ms_y_size = 1024
            pre_pan_x_size = 4096
            pre_pan_y_size = 4096
            # handle the last block (if the width or height is not proportional to the block size)
            if x_number == pan_nx_num - 1:
                pre_ms_x_size = (m_width - 1) % ms_block_size + 1
                pre_pan_x_size = (p_width - 1) % pan_block_size + 1
            if y_number == pan_ny_num - 1:
                pre_ms_y_size = (p_height//4 - 1) % ms_block_size + 1
                pre_pan_y_size = (p_height - 1) % pan_block_size + 1
            # read the block ms and pan image
            # ms_block_image = MS.ReadAsArray(x_number * ms_block_size, y_number * ms_block_size, pre_ms_x_size, pre_ms_y_size)
            # pan_block_image = PAN.ReadAsArray(x_number * pan_block_size, y_number * pan_block_size, pre_pan_x_size, pre_pan_y_size)
            ms_block_image=MS[:,x_number*ms_block_size:pre_ms_x_size+x_number*ms_block_size,y_number * ms_block_size:y_number * ms_block_size+pre_ms_y_size]
            pan_block_image=PAN[x_number * pan_block_size:pre_pan_x_size+x_number * pan_block_size, y_number * pan_block_size:y_number * pan_block_size+pre_pan_y_size]
            # unsample the ms image to make the ms equal to the size of pan
            up_ms_block_image = unsample(ms_block_image, factor)
            # pan_sharp(up_ms_block_image,pan_block_image,fusion_image)
            fusion_image[:4, y_number * pan_block_size: y_number * pan_block_size + pre_pan_y_size, x_number * pan_block_size:x_number * pan_block_size + pre_pan_x_size] \
                = pan_sharp(up_ms_block_image, pan_block_image)
    write_tiff(fusion_image, p_width, p_height, band_num, save_path)


if __name__ == '__main__':
    ms_path = "FullRegisterMS.tif"
    pan_path = "GF2_PMS1_E117.5_N39.2_20171208_L1A0002831428-PAN1.tiff"
    save_path = "result_pansharp.tif"
    fusion_main()