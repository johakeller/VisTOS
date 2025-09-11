'''
Module to download Sentinel-1 and Sentinel-2 satellite time series data to create 
Cameroon Deforestation-Driver Segmentation (CDDS) dataset, based on the "Labelled 
dataset to classify direct deforestation drivers in Cameroon" by Debus et al. (2024).
To run the acquisition, uncomment main() in the last line and run this script.

Credit for base dataset: Debus et al., 2024, [https://zenodo.org/records/8325259]
'''

import os
import shutil
import time

import cv2
import ee
import numpy as np
import pandas as pd
import rasterio
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

import params


def get_region_bounds(lat, lon, spat_extent=params.CDDS_IMG_WIDTH):
    '''
    Creates a rectangular bounding box given the center point of an 
    image in latitude/longitude and returns the bounding box as 
    ee.Geometry.Rectangle.
    '''

    # image gets one pixel too large otherwise: (333x333) 
    # -> should be (332x332)
    spat_extent_custom=(spat_extent-1)*params.CDDS_SCALE
    # spatial extent of bounding box in meters
    side_len=spat_extent_custom/2

    # get half side length in degree
    side_len_lat=side_len/params.FACTOR_METERS_PER_DEG
    side_len_lon=side_len/(params.FACTOR_METERS_PER_DEG*np.cos(np.deg2rad(lat)))

    return ee.Geometry.Rectangle([lon-side_len_lon, lat-side_len_lat, lon+side_len_lon, lat+side_len_lat])

def get_s1_data(start,end, coord_bounds):
    '''
    Function to fetch Sentinel-1 data for a given temporal extent and 
    given coordinate bounds (bounding box).
    '''

    # obtain Sentinel 1 data for each timestep
    s1_data = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterBounds(coord_bounds)\
        .filterDate(start, end)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))

    # only if data available
    if s1_data.size().getInfo() >0:
        # get mean for timestep
        s1_mean_time= s1_data.select(['VV','VH']).mean()
        # resample to 15m resolution 
        s1_clipped=s1_mean_time.clipToBoundsAndScale(coord_bounds, scale=params.CDDS_SCALE)
        s1_mean_resampled= s1_clipped.resample('bilinear').reproject(crs='EPSG:4326', scale=params.CDDS_SCALE)
        
        return s1_mean_resampled
  
def get_s2_data(start,end, coord_bounds):
    '''
    Function to fetch Sentinel-2 data for a given temporal extent and 
    given coordinate bounds (bounding box).
    '''

    # obtain Sentinel 2 data for each timestep
    s2_data = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterBounds(coord_bounds)\
        .filterDate(start, end)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35)) # cloud cover < 35%
    
    # only if data available
    if s2_data.size().getInfo() >0:
        # get image with lowest cloud cover per quarter
        s2_images=s2_data.filterDate(start, end).select(['B2','B3','B4','B8','B11','B12']) # blue, green, red, nir, swir1, swir2
        # sort by cloud cover in ascending order and pick the first
        s2_images_sorted = s2_images.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        # resample to 15m resolution 
        s2_clipped= s2_images_sorted.clipToBoundsAndScale(coord_bounds, scale=params.CDDS_SCALE)
        s2_resampled= s2_clipped.resample('bilinear').reproject(crs='EPSG:4326', scale=params.CDDS_SCALE)

        return s2_resampled

def get_global_forest_change(year, coord_bounds):
    '''
    Function to fetch Hansen Global Forest Change v1.11 (2000-2023) data for a 
    given year and coordinate bounds (bounding box).
    '''

    # obtain Global Forest Change data for the forest loss year
    gfc_data = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')\
        .clip(coord_bounds)\

    # get mask for loss year and previous year in correct format
    prev_year_format=ee.Number(float(year-1)).mod(2000)
    cur_year_format=ee.Number(float(year)).mod(2000)
    loss_year=gfc_data.select('lossyear')  
    # select mask for year:  0 (no loss) or else a value in the range 1-23, 
    # representing loss detected primarily in the year 2001-2023
    prev_loss_year=loss_year.eq(prev_year_format)  
    cur_loss_year=loss_year.eq(cur_year_format) 
    # combine 3 years to one mask
    label_region =prev_loss_year.max(cur_loss_year)
    
    return label_region

def get_sat_data(coord_bounds, sample_path, year):
    '''
    Function to acquire Sentinel-1 and Sentinel-2 satellite images, as well
    as Global Forest Change label maps, in the bands indicated in the 
    called corresponding functions, from Google Earth Engine. Downloads up 
    to 8 time steps over two years (quarters) to Google Drive and returns 
    started download tasks as list.
    '''    
    
    # get sample name
    sample_name=sample_path.replace('my_examples_landsat_final/','')

    # 8 timesteps from 2 years, forest loss year and year before
    temp_extent=[
        (ee.Date(f'{year-1}-01-01'), ee.Date(f'{year-1}-03-31')),
        (ee.Date(f'{year-1}-04-01'), ee.Date(f'{year-1}-06-30')),
        (ee.Date(f'{year-1}-07-01'), ee.Date(f'{year-1}-09-30')),
        (ee.Date(f'{year-1}-10-01'), ee.Date(f'{year-1}-12-31')),
        (ee.Date(f'{year}-01-01'), ee.Date(f'{year}-03-31')),
        (ee.Date(f'{year}-04-01'), ee.Date(f'{year}-06-30')),
        (ee.Date(f'{year}-07-01'), ee.Date(f'{year}-09-30')),
        (ee.Date(f'{year}-10-01'), ee.Date(f'{year}-12-31')),
        ]

    # save tasks in list to check their status    
    tasks_list=[]

    # get Global Forest Change label for year
    region_label=get_global_forest_change(year, coord_bounds)

    # export all bands for timestep to Google Drive
    region_label_export=ee.batch.Export.image.toDrive(
        image=region_label,
        description=f'region_label_{sample_name}_{year}', 
        fileNamePrefix=f'region_label_{sample_name}_{year}',
        region=coord_bounds,
        scale=params.CDDS_SCALE,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    # start export
    region_label_export.start()
    print(f'\rExport region label {sample_name}, year {year} started{params.EOL_SPACE}', end='')
    tasks_list.append( region_label_export)

    # iterate through timesteps
    for (start, end) in temp_extent: 
        # get month and year
        month=int(start.format('MM').getInfo())
        year=int(start.format('YYYY').getInfo())

        # obtain Sentinel 1 data for each timestep
        s1_sample=get_s1_data(start, end, coord_bounds)
        # returns None if no data available
        if s1_sample is not None:

            # export all bands for timestep to Google Drive
            s1_export=ee.batch.Export.image.toDrive(
                image=s1_sample,
                description=f's1_{sample_name}_{year}_{month}', 
                fileNamePrefix=f's1_{sample_name}_{year}_{month}',
                region=coord_bounds,
                scale=params.CDDS_SCALE,
                crs='EPSG:4326',
                maxPixels=1e13
            )
            # start export
            s1_export.start()
            print(f'\rExport sample {sample_name} S1, year {year}, month {month} started{params.EOL_SPACE}', end='')
            tasks_list.append(s1_export)
        
        # obtain Sentinel 2 data for each timestep
        s2_sample = get_s2_data(start, end, coord_bounds)
        # returns None if no data available
        if s2_sample is not None:

            # export all bands for timestep to Google Drive
            s2_export=ee.batch.Export.image.toDrive(
                image=s2_sample,
                description=f's2_{sample_name}_{year}_{month}', 
                fileNamePrefix=f's2_{sample_name}_{year}_{month}',
                region=coord_bounds,
                scale=params.CDDS_SCALE,
                crs='EPSG:4326',
                maxPixels=1e13
            )
            # start export
            s2_export.start()
            print(f'\rExport sample {sample_name} S2, year {year}, month {month} started{params.EOL_SPACE}', end='')
            tasks_list.append(s2_export)

    return tasks_list

def download_data_drive(drive_file, drive, local_sample_path):
    '''
    Downloads the GeoTIFF from Google Drive to local device 
    and deletes it to save memory. 
    '''

    # check if local directory exists
    if not os.path.exists(local_sample_path):
        os.makedirs(local_sample_path, exist_ok=True)

    # fetch from folder all files containing drive_file -> output list
    query= f"title contains '{drive_file}' and trashed=false"
    drive_files=drive.ListFile({'q': query}).GetList()

    # download all files from list
    for file in drive_files:
        # new local file name
        fn=file['title'].split('_')
        new_file_name='_'.join([fn[0], fn[-2], fn[-1]])
        file.GetContentFile(os.path.join(local_sample_path, new_file_name))
        print(f'\rDownloaded {file["title"]}{params.EOL_SPACE}', end='')
        # delete after download to save memory on Google Drive
        try:
            file.Delete()
            print(f'\rDeleted {file["title"]} from Google Drive{params.EOL_SPACE}', end='')
        except Exception as e:
            print(f'Google error deleting {file["title"]}{params.EOL_SPACE}',e, end='')

def normalize(channel, upper_bound=98, lower_bound=2):
    '''
    Applies normalization and gamma correction to multi-spectral satellite data.
    '''
    # replace nan values by 0
    values = np.nan_to_num(channel).astype(np.float32)
    # scale values
    values*=params.CDDS_S2_NORM_FACTOR
    # get percentile
    p_lower, p_upper=np.percentile(values, (lower_bound, upper_bound))
    # remove outliers
    values_clean=np.clip(values, p_lower,p_upper)
    # normalize (avoid 0-division)
    values_norm=(values_clean-p_lower)/max(1e-8,(p_upper-p_lower))
    # gamma correction
    gamma=1.8
    values_gamma=values_norm**(1/gamma)

    return values_gamma

def normalize_sar(channel, upper_bound=98, lower_bound=-25):
    '''
    Applies normalization to SAR data.
    '''
    # replace nan values by small value
    values = np.nan_to_num(channel, nan=1e-12).astype(np.float32)
    # get percentile
    p_upper=np.percentile(values, upper_bound)
    # normalize (avoid 0-division)
    values_norm=(values -lower_bound)/max(1e-8,(p_upper-lower_bound))

    return values_norm

def morph_mask(mask):
    '''
    Applies morphological closing to mask.
    '''

    mask=(mask>0).astype(np.uint8) *255 
    kernel_close=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    # morphological closing                            
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel_close)

    return (mask>0).astype(np.uint8)

def img_process(local_folder):
    '''
    Loads a GeoTIFF satellite image, applies preprocessing and converts it into .npy
    to save it on disk.
    '''

    processed_imgs=[]
    # iterate through all tifs in the directory
    for file in os.listdir(local_folder):
        if file.endswith('.tif'):
            file_path=os.path.join(local_folder,file)
            # open tif
            with rasterio.open(file_path) as band_data:
                # convert to numpy
                data_array=band_data.read()
                band_len=data_array.shape[0]

                # Sentinel-2 data has 6 bands
                if band_len==6:
                    # normalize Sentinel-2 data
                    bands_norm=normalize(data_array)
                # Sentinel-1 has 2 bands
                elif band_len==2:
                    bands_norm=normalize_sar(data_array)
                # else region label
                else:
                    bands_norm=morph_mask(data_array)
                # crop to 332x332 in case size does not match exactly
                bands_crop=bands_norm[:,:params.CDDS_IMG_WIDTH,:params.CDDS_IMG_WIDTH]
                # pad to 332x332 with 0 if smaller
                bands_pad=np.zeros((bands_crop.shape[0],params.CDDS_IMG_WIDTH,params.CDDS_IMG_WIDTH))
                bands_pad[:bands_crop.shape[0],:bands_crop.shape[1],:bands_crop.shape[2]]=bands_crop
                # save locally
                file_np_path= file_path.replace('.tif', '')
                # region_label
                if 'region' in file_np_path:
                    bands_pad=np.squeeze(bands_pad, axis=0)
                    file_np_path=os.path.join (local_folder, 'region_label.npy')
                np.save(file_np_path, bands_pad)
                print(f'\rSaved {file} as .npy{params.EOL_SPACE}', end='')
                processed_imgs.append((bands_pad, file))
            # remove .tif to save space
            os.remove(file_path)

def run_acquisition(queue_size=32, num_samples=1600):
    '''
    Starts satellite image (Sentinel-1, Sentinel-2), as well as Global 
    Forest Change maps acquisition with Google Earth Engine by using 
    queuing system. Reads the time steps and positions to be downloaded
    from the indicated .csv file from the dataset by Debus et al. (2024).
    As intermediate step, exports data to Google Drive to be downloaded from
    there to the local machine for further processing.
    '''
    
    # initialize Earth Engine
    ee.Authenticate()
    ee.Initialize()

    # initialize Pydrive
    google_auth=GoogleAuth()
    google_auth.LocalWebserverAuth()
    google_drive = GoogleDrive(google_auth)

    # read from CameroonDeforestationDrivers dataset
    data_path=params.CAMEROON_LABELS_PTH

    data_frame=pd.read_csv(data_path)

    # store new data per row in .csv file
    new_rows=[]
    total_rows=len(data_frame)
    row_counter=0
    tasks_queue=[]
    gen_samples=0

    # while rows or queued tasks left, limited by num_samples
    while (row_counter < total_rows or tasks_queue) and gen_samples<num_samples:
        # append rows to waiting queue
        while row_counter < total_rows and len(tasks_queue) < queue_size:
            row=data_frame.iloc[row_counter]
            # increase row count
            row_counter +=1
            # use Sentinel-2 Harmonized (started 2017) 
            if (row['year'] < 2018):
                continue
            
            sample_path=row['example_path']
            year=row['year']
            lat=row['latitude']
            lon=row['longitude']
            # get sample name
            sample_name=sample_path.replace('my_examples_landsat_final/','')
            local_folder_path=os.path.join(params.CDDS_PATH, f'samples/{sample_name}')
            
            # check if folder for sample already exists
            if not os.path.exists(local_folder_path):
                # get equal-sized coordinate bounding box
                coord_bounds = get_region_bounds(lat ,lon)
                # obtain satellite data
                task_list=get_sat_data(coord_bounds, sample_path, year)
                # append row as dictionary and corresponding tasks to a queue of running tasks
                for task in task_list:
                    tasks_queue.append((row.to_dict(), task))
            else:               
                # if folder exists write directly into new csv
                new_rows.append({
                    'label': row['label'],
                    'merged_label': row['merged_label'],
                    'latitude': lat, 
                    'longitude': lon,
                    'year': year,
                    'sample_path': os.path.join('samples',sample_name)
                })
                # count sample
                gen_samples+=1
        
        # append all tasks finished in the queue
        finished_tasks=[]
        for (row, task) in tasks_queue:
            if task.status()['state'] in ['COMPLETED', 'FAILED']:
                # append row as dictionary for removal later
                finished_tasks.append((row, task))

        # processed finished tasks
        for (row, task) in finished_tasks:
            # remove task from queue
            tasks_queue.remove((row, task))

            # read data
            sample_path=row['example_path']
            year=row['year']
            lat=row['latitude']
            lon=row['longitude']

            # get sample name
            sample_name=sample_path.replace('my_examples_landsat_final/','')
            sample_path_new='samples/'+sample_name
            local_folder_path=os.path.join(params.CDDS_PATH, sample_path_new)

            # download data from Google Drive
            download_data_drive(sample_name, google_drive, local_folder_path)
            # convert all GeoTIFFs in the local directory to .npy
            img_process(local_folder_path)

            # load srtm (only slope)
            srtm_path=os.path.join(params.CAMEROON_PATH, f'{sample_name}/auxiliary/slope.npy') 
            shutil.copy(srtm_path, os.path.join(local_folder_path,'srtm.npy'))

            # write into list
            new_rows.append({
                'label': row['label'],
                'merged_label': row['merged_label'],
                'latitude': lat, 
                'longitude': lon,
                'year': year,
                'sample_path': os.path.join('samples',sample_name)
            })
        # short break before next check
        time.sleep(10)
    # after all samples passed, save in new data frame, delete duplicates
    new_df=pd.DataFrame(new_rows).drop_duplicates()
    # save as .csv file
    csv_path = os.path.join(params.CDDS_PATH, 'all.csv')
    new_df.to_csv(csv_path, index=False, sep=',')
    print(f'\rSaved csv to {csv_path}{params.EOL_SPACE}', end='')

def main(): 
    '''
    Main() function to start the dataset acquisition pipeline.
    '''

    # start pipeline
    run_acquisition()

#main()
