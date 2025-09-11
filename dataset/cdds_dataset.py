'''
Module containing the Cameroon Deforestation -Driver Segmentation dataset class 
for fine-tuning. Calculation of inverse-frequency class weights for 
cross-entropy via main() method.

Credits for base dataset: Debus et al., 2024, [https://zenodo.org/records/8325259]
'''

import glob
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

import params


class CDDSDataset(IterableDataset):
    '''
    Dataset class for Cameroon Deforestation-Driver Segmentation (CDDS) dataset.
    '''

    def __init__(
        self, 
        split='train', 
        batch_size=params.FT_BATCH_SIZE, 
        max_length=None, 
        shuffle=False, 
        vis_field_size=params.VIS_FIELDS[0], 
        labels='merged', 
        seed=123
    ):
        # split defines which part of the dataset to load
        if split=='train':
            data_path=params.CDDS_TRAIN_PATH
        elif split=='test':
            data_path=params.CDDS_TEST_PATH
        elif split=='validation':
            data_path=params.CDDS_VAL_PATH
        else:
            raise NotImplementedError(f'Split {split} not implemented.')
        
        self.seed = seed
        self.rand=random.Random(self.seed)
        self.batch_size=batch_size
        # length of visual field side
        self.vis_field_size=vis_field_size
        self.padding=vis_field_size//2
        # list of names of sample directories for split
        try:
            self.data_frame=pd.read_csv(data_path)
        except FileNotFoundError as e:
            print(f'File {split}.csv must be in {params.CDDS_PATH}', e)
            sys.exit(1)
        self.sample_paths = self.data_frame['sample_path'].tolist()
        # use merged labels 
        self.label_type=labels
        self.labels=self.data_frame['merged_label'].tolist() if labels=='merged' else self.data_frame['label'].tolist() 
        self.lat=self.data_frame['latitude'].tolist()
        self.long=self.data_frame['longitude'].tolist()
        self.year=self.data_frame['year'].tolist()
        max_num_samples=len(self.sample_paths)*params.CDDS_NUM_PIXELS
        # check if enough samples available for desired length (in pixels) of the dataset
        if (max_length is not None) and (max_length > max_num_samples):
            raise IndexError(f'Length {max_length} is exceeding actual number of samples: ({max_num_samples})')
        else:
            # if parameter is set, limits the length of the dataset
            self.data_length=max_length if max_length is not None else max_num_samples
        # number of samples (images) used
        self.num_samples=self.data_length//params.CDDS_NUM_PIXELS
        # list of sample indices used
        self.sample_indices=list(range(len(self.sample_paths)))[:self.num_samples+1] 
        self.shuffle=shuffle
        if self.shuffle: 
            self.rand.shuffle(self.sample_indices)

    def __len__(self):
        '''
        Returns the dataset length in samples.
        '''

        return self.data_length
    
    def __iter__(self):
        '''
        Iterator method of the CDDS dataset. Iterates through the image time series of the dataset and inserts 
        the images at the corresponding time steps in an EO array. Iterates over the pixels of the EO array 
        and yields pixel time series as samples. Outputs the sample as a dictionary. 
        '''
        
        for idx, sample_index in enumerate(self.sample_indices):
            # all samples start in January before deforestation event
            start_month=0
            # get number of original pre-training input bands
            num_orig_bands= sum(values['length'] for keys,values in params.CHANNEL_GROUPS.items())+1 # include DW, include B9

            # create orig. sized array with 0s to represent missing bands, dimensions (num_orig_bands, 332, 332)
            eo_style_array = np.zeros([num_orig_bands, params.CDDS_MAX_SEQ_LEN,params.CDDS_IMG_WIDTH, params.CDDS_IMG_WIDTH])
            # indicate non-available channels with 1
            eo_style_mask = np.ones_like(eo_style_array)

            # prepare and pad the output data: loads the images from the dataset into their positions in EO array                  
            eo_style_array, eo_style_mask, eo_style_label, coords=self.prepare_channel_input(sample_index, eo_style_array, eo_style_mask)    

            # iterate over the pixels of the EO array and the corresponding mask and label
            for pix_id in range(params.CDDS_NUM_PIXELS):
                # check if reached max number of samples in last sample
                if idx == self.num_samples and pix_id >= (self.data_length%params.CDDS_NUM_PIXELS):
                    break
                # obtain pixel time series
                eo_data_vf=torch.tensor(eo_style_array[pix_id]).float().to(params.DEVICE)
                eo_mask_vf=torch.tensor(eo_style_mask[pix_id]).float().to(params.DEVICE)
                eo_label_vf=torch.tensor(eo_style_label[pix_id]).to(params.DEVICE)
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

    def init_class_weights(self):
        '''
        Loads the labeled regions to initiate inverse-frequency class weights based 
        on pixel count from the training dataset. Prints weights to console output
        and returns them as tensor.
        '''

        print(f'\rInitiating class weights.{params.EOL_SPACE}', end='')
        # convert to floats
        pixel_count={key: float(value) for key, value in params.CDDS_LABELS.items()}
        pixel_count['total']=0.0
        # set all 0
        for label in pixel_count.keys():
            pixel_count[label]=0

        # go through .csv file
        for index, label in enumerate(self.labels):
            # get name of the directory
            sample_id=self.sample_paths[index]
            # path of directory
            region_path=os.path.join(params.CDDS_PATH, f'{sample_id}/region_label.npy')
            # load region mask
            labeled_region=np.load(region_path)
            # total number of pixels
            total_pixels=labeled_region.shape[0]*labeled_region.shape[1]
            pixel_count['total']+= total_pixels
            sum_label=labeled_region.sum()
            pixel_count[label]+=sum_label
            pixel_count['No deforestation']+=total_pixels-sum_label

        # inverse-frequency weights
        for key, value in pixel_count.items():
            if key != 'total':
                pixel_count[key]=(1.0/value) if value != 0.0 else 0.0
        # print weights to console output
        print(pixel_count)
        
        class_weights= np.array(list(pixel_count.values()))
        # also returns weights as tensor
        return torch.tensor(class_weights, dtype=torch.float32).to(params.DEVICE)

    @staticmethod
    def calculate_coords(lat, lon):
        '''
        Fills a matrix size (CDDS_IMG_WIDTH, CDDS_IMG_WIDTH) with lat/lon coordinates,
        positioning the passed coordinates in the center of the image.
        '''
        coord_matrix = np.zeros((2, params.CDDS_IMG_WIDTH, params.CDDS_IMG_WIDTH), dtype=np.float32)
        # center of image
        center=params.CDDS_IMG_WIDTH//2
        for y in range(params.CDDS_IMG_WIDTH):
            for x in range(params.CDDS_IMG_WIDTH):
                # get offset from center of image (15 m/ pixel) for latitude
                coord_matrix[0, y, x]=((y-center)*15/111320) + lat
                # longitude offset (15 m/ pixel)
                coord_matrix[1,y,x]= ((x-center)*15/111320*math.cos(math.radians(lat)))+lon
        return coord_matrix                                

    @staticmethod
    def calc_month_index(image_path, event_year):
        '''
        Calculates the index of the month in the 24-months array of the interval 
        [01.year-1, 12.year], which is mapped on the indices in range [0,23].
        '''
        sample_year=int(os.path.basename(image_path).split('_')[1])
        # get month from file path (data is sample with lowest cloud cover of quarter, use start month here)
        file_name=os.path.basename(image_path).split('_')[2]
        month, _=os.path.splitext(file_name)
        month=int(month)
        # obtain index of month: if image is from event year
        if sample_year==event_year:
            month_idx=month+11
        # if the index of the month is from the previous year
        else:
            month_idx=month-1
        return month_idx

    def prepare_channel_input(
            self, 
            index, 
            eo_style_array, 
            eo_style_mask
        ):
        '''
        Helper method to extract EO data (in images) from the dataset and return 
        EO array, EO mask, labels and coordinates in the shapes of flattened 
        images in an EO array.
        '''

        # get sample directory name
        sample_id=self.sample_paths[index]

        # get the paths
        region_path=os.path.join(params.CDDS_PATH, f'{sample_id}/region_label.npy')
        srtm_path=os.path.join(params.CDDS_PATH, f'{sample_id}/srtm.npy')
        bands_path=os.path.join(params.CDDS_PATH, sample_id) 
        # get list of all Sentinel-2 images in directory
        s2_list=glob.glob(os.path.join(bands_path, 's2*'))

        # get list of Sentinel-1 data      
        s1_list=glob.glob(os.path.join(bands_path, 's1*'))
        event_year=self.year[index]
    
        if len(s2_list)>0:
            # load Sentinel-2
            for s2_image_path in s2_list:
                try:
                    s2_band_data=np.load(s2_image_path)
                    # get month index
                    s2_month_idx=self.calc_month_index(s2_image_path, event_year)
                    # insert rgb (first 3 bands)
                    eo_style_array[list(params.CHANNEL_GROUPS['S2_RGB']['idx']), s2_month_idx]=s2_band_data[:3]
                    # update EO mask
                    eo_style_mask[list(params.CHANNEL_GROUPS['S2_RGB']['idx']), s2_month_idx]=0
                    # insert NIR (4. band)
                    eo_style_array[list(params.CHANNEL_GROUPS['S2_NIR_10']['idx']), s2_month_idx]=s2_band_data[3]
                    eo_style_mask[list(params.CHANNEL_GROUPS['S2_NIR_10']['idx']), s2_month_idx]=0
                    # insert swir (5.,6. band), short wave infrared (index after index 10)
                    eo_style_array[[11,12], s2_month_idx]=s2_band_data[4:]
                    eo_style_mask[[11,12], s2_month_idx]=0
                    # calculate NDVI from NIR and red 
                    red_band = s2_band_data[2]
                    nir_norm= s2_band_data[3]
                    # avoid zero division
                    ndvi=np.zeros_like(red_band, dtype=np.float32)
                    denominator=nir_norm+red_band
                    np.divide(nir_norm -red_band, denominator, out=ndvi, where=denominator!=0)
                    # insert into EO-style array (comes after later removed index 10, so 16+1)
                    eo_style_array[17, s2_month_idx]=ndvi
                    # update EO mask
                    eo_style_mask[17, s2_month_idx]=0
                except ValueError as e:
                    print(f'Sample {sample_id} S2 data contains pickled data: {e}')
                
        if len(s1_list)>0:
            # load Sentinel-1 images 
            for s1_image_path in s1_list:
                try:
                    s1_band_data=np.load(s1_image_path)
                    # get month index -> possibly not same as Sentinel-2 (different coverage)
                    s1_month_idx=self.calc_month_index(s1_image_path, event_year)
                    # insert into EO-style array
                    eo_style_array[list(params.CHANNEL_GROUPS['S1']['idx']), s1_month_idx]=s1_band_data
                    # update EO mask
                    eo_style_mask[list(params.CHANNEL_GROUPS['S1']['idx']), s1_month_idx]=0
                except ValueError as e:
                    print(f'Sample {sample_id} S1 data contains pickled data: {e}')

        slope=np.load(srtm_path)
        # insert into EO-style array, SRTM beyond B9 index(10)-> index+1
        # insert at the first timestep (0) only -> only this one used in embedding (time-independent)
        eo_style_array[[15],0]=slope
        # update EO mask
        eo_style_mask[[15],0]=0 

        # create matrix with coordinates for each pixel
        coords=self.calculate_coords(self.lat[index], self.long[index])

        # load region_label and multiply ones with class index
        region_label=np.load(region_path)*params.CDDS_LABELS[self.labels[index]]
 
        # remove band B9 (row index 10) from array and masks
        eo_style_array = np.delete(eo_style_array, 10, axis=0)
        eo_style_mask = np.delete(eo_style_mask, 10, axis=0) 

        # mask values (are ignored anyway)
        eo_style_array[:]=eo_style_array *np.logical_not(eo_style_mask.astype(bool))

        # (c, t, h,w) -> reshape and flatten (h*w,t,c)
        c,t,h,w=eo_style_array.shape
        eo_style_array=eo_style_array.reshape(c,t,h*w).transpose(2,1,0)
        eo_style_mask=eo_style_mask.reshape(c,t,h*w).transpose(2,1,0)
        # (h,w)-> (h*w)
        region_label=region_label.reshape(h*w)
        #(2,h,w)->(h*w,2)
        coords=coords.reshape(2,h*w).transpose(1,0)
       
        return eo_style_array, eo_style_mask, region_label, coords
    
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

        # input dictionary: collect channel groups in dictionary {band_name: numpy array}, labels and masks
        input_dict={}
        input_dict['EO']=eo_data
        input_dict['DW']=dw_data
        input_dict['loc']=lat_lon
        input_dict['EO_label']=eo_label
        input_dict['EO_mask']=eo_mask
        input_dict['DW_mask']=dw_mask
        input_dict['month']=start_months
        return input_dict

def main(): 
    '''
    Main function to be called for the calculation of inverse-frequency
    class weights.
    '''

    # calculate class weights
    ds = CDDSDataset()
    ds.init_class_weights()

#main()