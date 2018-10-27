import gdal
import cv2
MS_path=r'F:\GF2_Registration\GF2_PMS1_E113.3_N23.1_20170915_L1A0002600402-MSS1.tiff'
PAN_path=r'F:\GF2_Registration\GF2_PMS1_E113.3_N23.1_20170915_L1A0002600402-PAN1.tiff'
MS_dataset = gdal.Open(MS_path)
PAN_dataset = gdal.Open(PAN_path)
MS = MS_dataset.ReadAsArray(0, 0, MS_dataset.RasterXSize, MS_dataset.RasterYSize)
PAN = PAN_dataset.ReadAsArray(0, 0, PAN_dataset.RasterXSize, PAN_dataset.RasterYSize)
PAN=cv2.resize(PAN,dsize=(MS.shape[2],MS.shape[1]),interpolation=cv2.INTER_LINEAR)
driver=gdal.GetDriverByName('GTiff')
datasetMS = driver.Create('MS.tif', int(0.1*MS.shape[2]),int(0.1*MS.shape[1]), 4, gdal.GDT_UInt16)

for i in range(4):
    datasetMS.GetRasterBand(i+1).WriteArray(MS[i,0:int(0.1*MS.shape[1]),0:int(0.1*MS.shape[2])])

datasetPAN = driver.Create('PAN.tif',  int(0.1*MS.shape[2]),int(0.1*MS.shape[1]), 1, gdal.GDT_UInt16)
datasetPAN.GetRasterBand(1).WriteArray(PAN[0:int(0.1*MS.shape[1]),0:int(0.1*MS.shape[2])])