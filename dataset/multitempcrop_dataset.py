'''
Module containing the Cameroon Deforestation -Driver Segmentation dataset class 
for fine-tuning. Calculation of inverse-frequency class weights for 
cross-entropy via main() method.

Credits for base dataset: Debus et al., 2024, [https://zenodo.org/records/8325259]
'''
import os
import random
import sys

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio import transform as rtx
from rasterio import warp as rwarp
from torch.utils.data import IterableDataset

import params


class MultiTempCropClass(IterableDataset):
    '''
    Dataset class for BraDD-S1TS dataset.
    '''

    def __init__(
        self, 
        split='train', 
        batch_size=params.MTCC_BATCH_SIZE, 
        shuffle=False, 
        seed=123
    ):      
        self.seed = seed
        self.rand=random.Random(self.seed)
        self.batch_size=batch_size

        # split defines which part of the dataset to load
        if split=='train':
            data_path= os.path.join(params.MTCC_PATH, 'training_data.txt') 
            self.samples_path=os.path.join(params.MTCC_PATH, 'training_chips')
        elif split=='test':
            data_path= os.path.join(params.MTCC_PATH, 'test_data.txt') 
            self.samples_path=os.path.join(params.MTCC_PATH, 'test_chips')
        elif split=='validation':
           data_path= os.path.join(params.MTCC_PATH, 'validation_data.txt') 
           self.samples_path=os.path.join(params.MTCC_PATH, 'validation_chips')
        else:
            raise NotImplementedError(f'Split {split} not implemented.')
        
        # get split samples idx list
        try:
            with open(data_path, 'r', encoding='utf-8') as file:
                self.sample_idx= [row.strip() for row in file]
            if shuffle: 
                self.rand.shuffle(self.sample_idx)

        except FileNotFoundError as e:
            if split == 'train':
                print(f'File training_data.txt must be in {params.MTCC_PATH}', e)
            else:
                print(f'File {split}_data.txt must be in {params.MTCC_PATH}', e)
            sys.exit(1)

        # get corresponding metadata
        chips_df_path= os.path.join(params.MTCC_PATH, 'chips_df.csv') 
        try:
            chips_df=pd.read_csv(chips_df_path).set_index("chip_id", drop=False)
            # metadata dictionary
            self.meta=chips_df.to_dict(orient='index')
        except FileNotFoundError as e:
            if split == 'train':
                print(f'File chips_df.csv must be in {params.MTCC_PATH}', e)
            sys.exit(1)

        # number of samples (images) used
        self.num_samples=len(self.sample_idx)
        self.data_length=self.num_samples*params.MTCC_NUM_PIXELS
        # get number of original pre-training input bands
        self.num_bands= sum(values['length'] for values in params.CHANNEL_GROUPS.values()) # exclude DW, include B9

    def __len__(self):
        '''
        Returns the dataset length in samples.
        '''

        return self.data_length
    
    def __iter__(self):
        '''
        Iterator method of the BraDD-S1TS dataset. Iterates through the image time series of the dataset and inserts 
        the images at the corresponding time steps in an EO array. Iterates over the pixels of the EO array 
        and yields pixel time series as samples. Outputs the sample as a dictionary. 
        '''
        
        for idx, sample_id in enumerate(self.sample_idx):

            # create orig. sized array with 0s to represent missing bands, dimensions (num_orig_bands, 332, 332)
            eo_style_array = torch.zeros([self.num_bands, params.MTCC_MAX_SEQ_LEN,params.MTCC_IMG_WIDTH, params.MTCC_IMG_WIDTH])
            # indicate non-available channels with 1
            eo_style_mask = torch.ones_like(eo_style_array)

            # prepare and pad the output data: loads the images from the dataset into their positions in EO array                  
            eo_style_array, eo_style_mask, eo_style_label, coords, start_month=self.prepare_channel_input(sample_id, eo_style_array, eo_style_mask)    

            # iterate over the pixels of the EO array and the corresponding mask and label
            for pix_id in range(params.MTCC_NUM_PIXELS):
                # check if reached max number of samples in last sample
                if idx == self.num_samples and pix_id >= (self.data_length%params.MTCC_NUM_PIXELS):
                    break
                # obtain pixel time series
                eo_data_vf=eo_style_array[pix_id].float().to(params.DEVICE)
                eo_mask_vf=eo_style_mask[pix_id].float().to(params.DEVICE)
                eo_label_vf=eo_style_label[pix_id].to(params.DEVICE)
                dw_data_vf=torch.zeros(eo_mask_vf.shape[0],dtype=torch.float32).to(params.DEVICE)
                dw_mask_vf=torch.ones_like(dw_data_vf).to(params.DEVICE)
                coords_data_vf=torch.tensor(coords[pix_id]).float().to(params.DEVICE)
                start_months_vf=start_month
                
                # yield input dictionary
                yield self.create_input_dict(
                    eo_data=eo_data_vf,
                    dw_data=dw_data_vf,
                    lat_lon=coords_data_vf,
                    start_months=start_months_vf,
                    eo_mask=eo_mask_vf,
                    eo_label=eo_label_vf,
                    dw_mask=dw_mask_vf
                )  
    
    def get_month_idx(self, sample_id):
        '''
        Calculates the index of the month in the 24-months array of the interval 
        [01.year-1, 12.year], which is mapped on the indices in range [0,23].
        '''
        sample_meta=self.meta[sample_id]
        # get start month index
        start_month= (pd.to_datetime(sample_meta['first_img_date']).to_period('M')).month-1
        # get middle month index
        mid_month= (pd.to_datetime(sample_meta['middle_img_date']).to_period('M')).month-1
        # get last month index
        last_month= (pd.to_datetime(sample_meta['last_img_date']).to_period('M')).month-1

        return [start_month, mid_month, last_month]
       
    @staticmethod
    def generate_coords(transform, src_crs, dst_crs=params.LATLON_CRS):
        '''
        Fills a matrix size (BRADD_IMG_WIDTH, BRADD_IMG_WIDTH) with lat/lon coordinates,
        positioning the randomly generated coordinates from a passed range in the 
        center of the image.
        '''
        width=params.MTCC_IMG_WIDTH
        height=params.MTCC_IMG_WIDTH
        # get rows and column matrices
        rows, cols=np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        # get coordinates from affine transform
        x, y=rtx.xy(transform, rows, cols, offset="center")
        x, y = np.asarray(x), np.asarray(y)
        # convert to lat/lon
        lon, lat=rwarp.transform(src_crs=src_crs, dst_crs=dst_crs, xs=x.ravel(), ys=y.ravel())
        lat=np.asarray(lat).reshape(height,width)
        lon=np.asarray(lon).reshape(height, width)
        # get lat, lon pairs into on matrix
        return np.stack([lat, lon], axis=0).astype(np.float32)
    
    def prepare_channel_input(
            self, 
            sample_id, 
            eo_tensor, 
            eo_mask
        ):
        '''
        Helper method to extract EO data (in images) from the dataset and return 
        EO array, EO mask, labels and coordinates in the shapes of flattened 
        images in an EO array.
        '''

        # EO data path
        eo_path=os.path.join(self.samples_path, f'{sample_id}_merged.tif')
        # load data
        with rasterio.open(eo_path) as src:
            eo_data=src.read()
            eo_transform=src.transform
            eo_crs=src.crs
        # (t*c, h,w)
        eo_data=torch.from_numpy(eo_data.astype(np.float32, copy=False))
        # normalize to range [0,1]
        eo_data=eo_data*params.MTCC_NORM
        # get into shape (c,t, h, w)
        eo_data=eo_data.reshape(params.MTCC_TIME_STEPS,params.MTCC_NUMBER_CHANNELS,params.MTCC_IMG_WIDTH, params.MTCC_IMG_WIDTH)
        eo_data=eo_data.permute(1,0,2,3)

        # label data path
        label_path=os.path.join(self.samples_path, f'{sample_id}.mask.tif')
        # load label
        with rasterio.open(label_path) as src:
            label_data=src.read(1)
        label_data=torch.from_numpy(label_data).long()

        # get month index -> possibly not same as Sentinel-2 (different coverage)
        month_idx=self.get_month_idx(sample_id)
        start_month=month_idx[0]

        # insert into EO-style tensor -> select the correct channel groups
        # RGB
        rgb_idx=torch.as_tensor(list(params.CHANNEL_GROUPS['S2_RGB']['idx']))
        src_rgb_idx=params.MTCC_CHANNEL_GROUPS['S2_RGB']
        # fill in data
        eo_tensor[rgb_idx[:,None], month_idx]=eo_data[src_rgb_idx]
        # update EO mask
        eo_mask[rgb_idx[:,None],month_idx]=0

        # NIR
        nir_idx=torch.as_tensor(list(params.CHANNEL_GROUPS['S2_NIR_20']['idx']))
        src_nir_idx=params.MTCC_CHANNEL_GROUPS['S2_NIR_20']
        # fill in data
        eo_tensor[nir_idx[:,None], month_idx]=eo_data[src_nir_idx]
        # update EO mask
        eo_mask[nir_idx[:,None],month_idx]=0

        # SWIR
        swir_idx=torch.as_tensor(list(params.CHANNEL_GROUPS['S2_SWIR']['idx']))
        src_swir_idx=params.MTCC_CHANNEL_GROUPS['S2_SWIR']
        # fill in data
        eo_tensor[swir_idx[:,None], month_idx]=eo_data[src_swir_idx]
        # update EO mask
        eo_mask[swir_idx[:,None],month_idx]=0

        # NDVI from NIR and red ((NIR-red)/(nir+red))
        ndvi_idx=torch.as_tensor(list(params.CHANNEL_GROUPS['NDVI']['idx']))
        red_band = eo_data[2]
        nir_band= eo_data[3]
        # avoid zero division
        ndvi=torch.zeros_like(red_band, dtype=torch.float32)
        denominator=nir_band+red_band
        ndvi=torch.where(denominator !=0, (nir_band-red_band)/denominator, ndvi)
        # fill in data
        eo_tensor[ndvi_idx[:,None], month_idx]=ndvi
        # update EO mask
        eo_mask[ndvi_idx[:,None],month_idx]=0


        # create matrix with coordinates for each pixel based on random center point
        coords=self.generate_coords(eo_transform, eo_crs)

        # mask values (are ignored anyway)
        eo_tensor=eo_tensor.masked_fill(eo_mask.to(dtype=torch.bool),0) 

        # (c, t, h,w) -> reshape and flatten (h*w,t,c)
        c,t,h,w=eo_tensor.shape
        eo_tensor=eo_tensor.reshape(c,t,h*w).permute(2,1,0)
        eo_mask=eo_mask.reshape(c,t,h*w).permute(2,1,0)
        # (h,w)-> (h*w)
        region_label=label_data.reshape(h*w)
        #(2,h,w)->(h*w,2)
        coords=coords.reshape(2,h*w).transpose(1,0)
            
        return eo_tensor, eo_mask, region_label, coords, start_month
    
    def create_input_dict(
        self, 
        eo_data, 
        dw_data,
        lat_lon, 
        start_months, 
        eo_mask=None, 
        eo_label=None, 
        dw_mask=None
    ):
        '''
        Returns a dictionary of channel groups, masks and labels, as well as a start month for 
        a pixel time series sample as the input for the VisTOS model.
        '''

        input_dict={}
        input_dict['EO']=eo_data
        input_dict['DW']=dw_data
        input_dict['loc']=lat_lon
        input_dict['EO_label']=eo_label
        input_dict['EO_mask']=eo_mask
        input_dict['DW_mask']=dw_mask
        input_dict['month']=start_months
        return input_dict



# test
#ds = MulTempCrop()
#it = iter(ds)
#sample = next(it)


