'''
Module containing the PASTIS-R dataset class for fine-tuning. 
Calculation of inverse-frequency class weights for 
cross-entropy loss via main() method.

Credits: Sainte Fare Garnot et al., 2022, https://github.com/VSainteuf/pastis-benchmark/blob/main/code/dataloader.py#L26
'''

import json
import os
import random
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from rasterio.transform import from_bounds, xy
from torch.utils.data import DataLoader, IterableDataset

import params


class PastisRDataset(IterableDataset):
    '''
    Dataset class for the PASTIS-R semantic segmentation dataset.
    '''

    def __init__(
        self, 
        split='train', 
        batch_size=params.P_BATCH_SIZE, 
        max_length=None, 
        shuffle=False, 
        seed=123
    ):
        self.seed = seed
        self.rand=random.Random(self.seed)
        self.batch_size=batch_size
    
        # get metadata and sort samples by index
        self.meta_data=gpd.read_file(params.P_METADATA)
        self.meta_data.index=self.meta_data['ID_PATCH'].astype(int)
        self.meta_data.sort_index(inplace=True)

        # create dictionary of date tables for each satellite image type counting in days from
        # # reference date and indicating at which days images are available
        self.sats=['S2', 'S1A','S1D']
        # start date
        self.reference_date = datetime(*map(int, '2018-09-01'.split("-")))
        # dictionary for availability tables of satellite data
        self.date_tables = {s: None for s in self.sats}
        # time range in days relative to reference
        self.date_range = np.array(range(-200, 600))
        # create binary availability table for image data for each satellite
        for s in self.sats:
            # dates from metadata 
            dates = self.meta_data["dates-{}".format(s)]
            # date table: rows: indices, columns: dates relative to reference
            date_table = pd.DataFrame(
                index=self.meta_data.index, columns=self.date_range, dtype=int
            )
            # for sequence of dates for image data
            for pid, date_seq in dates.items():
                if type(date_seq) == str:
                    date_seq = json.loads(date_seq)
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                # convert YYYYMMDD to datetime
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                # mark up available dates (with imagery) with 1
                date_table.loc[pid, d.values] = 1
            # everything else 0
            date_table = date_table.fillna(0)
            # save as dictionary for each satellite
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }
        # define split by fold
        if split =='train':
            folds=[1,2,3]
        elif split =='val':
            folds=[4]
        # test
        else:
            folds=[5]

        # create dataset from selected folds
        self.meta_data = pd.concat(
                [self.meta_data[self.meta_data["Fold"] == f] for f in folds]
            )
        
        # number of patches
        self.num_patches=self.meta_data.shape[0]
        # define number of pixels as length 
        self.data_length=self.num_patches*params.P_NUM_PIXELS 
        # patch indices
        self.patch_ids=list(self.meta_data.index)
            
        # check if enough samples available for desired length (in pixels) of the dataset
        if (max_length is not None):
            if (max_length > self.data_length):
                raise IndexError(f'Length {max_length} is exceeding actual number of samples ({self.data_length})')
            else:
                # if parameter is set, limits the length of the dataset
                self.data_length=max_length 
                # truncate to corresponding number of patches
                self.patch_ids=self.patch_ids[:max_length//params.P_NUM_PIXELS]
        
        # apply shuffling
        if shuffle: 
            self.rand.shuffle(self.patch_ids)

        # Get normalization values
        self.norm = {}
        for s in self.sats:
            with open(os.path.join(params.P_PATH, 'NORM_{}_patch.json'.format(s)), 'r') as file:
                normvals = json.loads(file.read())
            selected_folds = folds if folds is not None else range(1, 6)
            means = [normvals['Fold_{}'.format(f)]['mean'] for f in selected_folds]
            stds = [normvals['Fold_{}'.format(f)]['std'] for f in selected_folds]
            self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
            self.norm[s] = (
                # mean
                self.norm[s][0],
                # standard deviation
                self.norm[s][1],
            )

    def __len__(self):
        '''
        Returns length of the dataset.
        '''
        
        return self.data_length
    
    def __iter__(self):
        '''
        Iterator method of the PASTIS-R dataset. Iterates through the image time series of the dataset and inserts 
        the images at the corresponding time steps in an EO array. Iterates over the pixels of the EO array and 
        yields pixel time series as samples. Outputs the sample as a dictionary. 
        '''

        for idx, item in enumerate(self.patch_ids):
            # all samples start September (see reference date)
            start_month=9

            # get number of original pre-training input bands
            num_orig_bands= sum(values['length'] for keys,values in params.CHANNEL_GROUPS.items())+1 # include DW, include B9

            # create orig. sized array with 0s to represent missing bands, dimensions (num_orig_bands, t=P_MAX_SEQ_LEN, 128, 128)
            eo_style_array = np.zeros([num_orig_bands, params.P_MAX_SEQ_LEN, params.P_IMG_WIDTH, params.P_IMG_WIDTH])
            # indicate unavailable channels with 1
            eo_style_mask = np.ones_like(eo_style_array)

            # prepare and pad the output data                    
            eo_style_array, eo_style_mask, eo_style_label, coords=self.prepare_channel_input(item, eo_style_array, eo_style_mask)    

            # iterate over pixels
            for pix_id in range(params.P_NUM_PIXELS):
                # check if reached max number of samples in last sample
                if (idx == len(self.patch_ids)-1) and pix_id >= (self.data_length%params.P_NUM_PIXELS):
                    break
                # obtain pixel time series
                eo_data=torch.tensor(eo_style_array[pix_id]).float().to(params.DEVICE)
                eo_mask=torch.tensor(eo_style_mask[pix_id]).float().to(params.DEVICE)
                # label is the first dimension of the np.array
                eo_label=torch.tensor(eo_style_label[0][pix_id], device=params.DEVICE, dtype=torch.long)
                dw_data=torch.zeros(eo_mask.shape[0],dtype=torch.float32).to(params.DEVICE)
                dw_mask=torch.ones_like(dw_data).to(params.DEVICE)
                coords_data=torch.tensor(coords[pix_id]).float().to(params.DEVICE)
                start_months=start_month
                
                # yield input dictionary
                yield self.create_input_dict(
                    eo_data=eo_data,
                    dw_data=dw_data,
                    lat_lon=coords_data,
                    start_months=start_months,
                    eo_mask=eo_mask,
                    eo_label=eo_label,
                    dw_mask=dw_mask
                )  

    @staticmethod
    def calculate_coords(coordinates):
        '''
        Fills a matrix  of size (IMG_LENGTH, IMG_LENGTH) with lat/lon coordinates
        assuming coordinates are passed as bounding box.
        '''

        # get the x (width) and y (height) extent
        x_left=coordinates[3][0]
        y_bottom=coordinates[2][1]
        x_right=coordinates[1][0]
        y_top=coordinates[0][1]

        # affine transform
        transform=from_bounds(x_left, y_bottom, x_right, y_top, params.P_IMG_WIDTH, params.P_IMG_WIDTH)

        # target matrix
        coord_matrix = np.zeros((2, params.P_IMG_WIDTH, params.P_IMG_WIDTH), dtype=np.float32)
        # iterate over pixels of target matrix to calculate their coords
        for y in range (params.P_IMG_WIDTH):
            for x in range(params.P_IMG_WIDTH):
                # get pixel coordinates in lat/lon
                lon, lat=xy(transform, y,x, offset='center')
                coord_matrix[0, y,x]=lat
                coord_matrix[1, y,x]=lon

        return coord_matrix                                

    def calc_monthly_median(
            self, 
            id_patch, 
            id_s, 
            band_data
        ):
        '''
        Given the band data for either S1A, S1D, or S2 for a certain patch, 
        calculates the monthly medians of the imagery and truncates the 
        time dimension to 12 consecutive months.
        '''

        # mean per month -> get datetable for the data (time difference to reference in days)
        valid_days=self.get_dates(id_patch, id_s)
        # get the real date 
        day_table=[self.reference_date+timedelta(days=int(day))for day in valid_days]

        # group by month
        months_grouped={}
        for index, date in enumerate(day_table):
            # group by (year, month)
            key=(date.year, date.month)
            if key in months_grouped:
                months_grouped[key].append(index)
            else:
                months_grouped[key]=[]
                months_grouped[key].append(index)
        
        # sort the kys chronologically
        months_grouped_sorted=sorted(months_grouped.keys())

        data_median = []
        for month in months_grouped_sorted:
            # select all valid time indices per month in (t, c, h, w)
            month_collection=band_data[months_grouped[month],:,:,:]
            # monthly median -> (c, h,w)
            month_median=np.median(month_collection, axis=0)
            data_median.append(month_median)

        # truncate to desired length
        data_median_list= data_median[:params.P_MAX_SEQ_LEN]
        return np.stack(data_median_list)
    
    def get_dates(self, id_patch, sat):
        '''
        Returns relative dates as table of days on which imagery is available 
        for id_patch and the satellite.
        '''

        return self.date_range[np.where(self.date_tables[sat][id_patch]==1)[0]]
                                        
    def normalize(
        self,
        band_data, 
        s_id, 
        plottable_norm=True
    ):
        '''
        Applies channel-wise normalization to each band group. Standard normalization yields
        values beyond interval [0,1], which is not sufficinet for plotting, additional plotting 
        normalization possible.
        '''        

        # standard PASTIS normalization -> not sufficient
        band_data_std_norm=(band_data - self.norm[s_id][0][None, :, None, None])/ self.norm[s_id][1][None, :, None, None]

        # further normalization for plottable colors
        if plottable_norm:
            band_data_plot_norm=np.zeros_like(band_data_std_norm)
            for chan_id in range (band_data.shape[0]):
                channel=band_data_std_norm[chan_id]
                chan_min=channel.min()
                chan_max=channel.max()
                # avoid zero division
                if chan_min < chan_max:
                    band_data_plot_norm[chan_id]=(channel-chan_min)/(chan_max-chan_min)
            return band_data_plot_norm
        return band_data_std_norm

    def prepare_channel_input(
        self, 
        id_patch, 
        eo_style_array, 
        eo_style_mask
    ):
        '''
        Helper method to extract EO data (in images) from the dataset and return 
        EO array, EO mask, labels and coordinates in the shapes of flattened 
        images in an EO array.
        '''
        
        # get the corresponding data
        # Sentinel-1 A (t, c, h, w)
        s1a_data=np.load(os.path.join(params.P_S1A_PATH, f'S1A_{id_patch}.npy')) # use only ascending
        # Sentinel 2 (t, c, h, w)
        s2_data=np.load(os.path.join(params.P_S2_PATH, f'S2_{id_patch}.npy'))
        # patch label (3, h, w)
        region_label = np.load(os.path.join(params.P_PATH, 'ANNOTATIONS', f'TARGET_{id_patch}.npy'))
        # load coordinates of 4 corner points as list of tuples
        geometry=self.meta_data.loc[id_patch]['geometry']
        if geometry.geom_type=='Polygon':
            coordinates=list(geometry.exterior.coords)
        elif geometry.geom_type=='MultiPolygon':
            coordinates=list(geometry.geoms[0].exterior.coords)
        else:
            raise ValueError(f'Geometry type {geometry.geom_type} not supported.')

        # median per month and truncate to 12 months
        s1a_data_median= self.calc_monthly_median(id_patch,'S1A' , s1a_data)
        s2_data_median= self.calc_monthly_median(id_patch,'S2', s2_data)

        # normalization and transformation from (t,c,h,w) to (c,t,h,w)
        s1a_data_norm=self.normalize(s1a_data_median, 'S1A').transpose(1,0,2,3) 
        #s1a_data_norm=s1a_data_median.transpose(1,0,2,3)
        s2_data_norm=self.normalize(s2_data_median, 'S2').transpose(1,0,2,3)
        #s2_data_norm=s2_data_median.transpose(1,0,2,3)

        # insert Sentinel-2
        # insert rgb (first 3 bands)
        eo_style_array[list(params.CHANNEL_GROUPS['S2_RGB']['idx']),:params.P_MAX_SEQ_LEN]=s2_data_norm[:3]
        # update EO mask
        eo_style_mask[list(params.CHANNEL_GROUPS['S2_RGB']['idx']), :params.P_MAX_SEQ_LEN]=0
        # insert Red Edge (bands 4-6)
        eo_style_array[list(params.CHANNEL_GROUPS['S2_Red_Edge']['idx']),:params.P_MAX_SEQ_LEN]=s2_data_norm[3:6]
        eo_style_mask[list(params.CHANNEL_GROUPS['S2_Red_Edge']['idx']), :params.P_MAX_SEQ_LEN]=0
        # insert NIR 10 
        eo_style_array[list(params.CHANNEL_GROUPS['S2_NIR_10']['idx']), :params.P_MAX_SEQ_LEN]=s2_data_norm[6]
        eo_style_mask[list(params.CHANNEL_GROUPS['S2_NIR_10']['idx']), :params.P_MAX_SEQ_LEN]=0
        # insert NIR 20
        eo_style_array[list(params.CHANNEL_GROUPS['S2_NIR_20']['idx']), :params.P_MAX_SEQ_LEN]=s2_data_norm[7]
        eo_style_mask[list(params.CHANNEL_GROUPS['S2_NIR_20']['idx']), :params.P_MAX_SEQ_LEN]=0


        # insert swir (5.,6. band), short wave infrared (index after index 10)
        eo_style_array[[11,12],  :params.P_MAX_SEQ_LEN]=s2_data_norm[8:]
        eo_style_mask[[11,12],  :params.P_MAX_SEQ_LEN]=0
        # calculate NDVI from NIR and red 
        red_band = s2_data_norm[2]
        nir_norm= s2_data_norm[6]
        ndvi=np.zeros_like(red_band, dtype=np.float32)
        denominator=nir_norm+red_band
        np.divide(nir_norm -red_band, denominator, out=ndvi, where=denominator!=0)
        # insert into eo-style array (comes after removed index 10, so 16+1)
        eo_style_array[17, :params.P_MAX_SEQ_LEN]=ndvi
        # update eo mask
        eo_style_mask[17, :params.P_MAX_SEQ_LEN]=0
                
        # insert Sentinel-1 
        eo_style_array[list(params.CHANNEL_GROUPS['S1']['idx']), :params.P_MAX_SEQ_LEN]=s1a_data_norm[:2]
        # update eo mask
        eo_style_mask[list(params.CHANNEL_GROUPS['S1']['idx']), :params.P_MAX_SEQ_LEN]=0

        # create matrix with coordinates for each pixel
        pixel_coords=self.calculate_coords(coordinates)
 
        # remove band B9 (row index 10) from array and masks
        eo_style_array = np.delete(eo_style_array, 10, axis=0)
        eo_style_mask = np.delete(eo_style_mask, 10, axis=0) 

        # mask values (are ignored anyway)
        eo_style_array[:]=eo_style_array *np.logical_not(eo_style_mask.astype(bool))

        # (c, t, h,w) -> reshape and flatten (h*w,t,c)
        c,t,h,w=eo_style_array.shape
        eo_style_array=eo_style_array.transpose(2,3,1,0)
        eo_style_array=eo_style_array.reshape(h*w,t,c)
        eo_style_mask=eo_style_mask.transpose(2,3,1,0)
        eo_style_mask=eo_style_mask.reshape(h*w,t,c)
        # (3,h,w) -> (3, h*w) 
        region_label=region_label.reshape(3, h*w)
        # (2,h,w) -> (h*w,2)
        pixel_coords=pixel_coords.reshape(2,h*w).transpose(1,0)
       
        return eo_style_array, eo_style_mask, region_label, pixel_coords
    
    def create_input_dict(
        self, 
        eo_data, 
        dw_data,lat_lon, 
        start_months, 
        eo_mask=None, 
        eo_label=None, 
        dw_mask=None
        ):
        '''
        Returns a dictionary of channel groups, masks and labels, as well as a start month for 
        a pixel time series sample as the input for the VisTOS model.
        '''

        # input dictionary: collect channel groups in dictionary {channel group: numpy array}, labels and masks
        input_dict={}
        input_dict['EO']=eo_data
        input_dict['DW']=dw_data.squeeze(-1)
        input_dict['loc']=lat_lon
        input_dict['EO_label']=eo_label
        input_dict['EO_mask']=eo_mask
        input_dict['DW_mask']=dw_mask.squeeze(-1)
        input_dict['month']=start_months
        return input_dict

def init_class_weights_pastis():
    '''
    Loads the labeled regions to initiate inverse-frequency class weights based 
    on pixel count from the training dataset. Prints weights to console output
    and returns them as tensor.
    '''

    print(f'\rInitiating class weights.{params.EOL_SPACE}', end='')

    # get train dataset
    train_ds = PastisRDataset(split='train',shuffle=False)
    
    # dataloader
    train_dl = DataLoader(
                train_ds,
                # 20 images per batch
                batch_size=params.P_NUM_PIXELS*20,
                num_workers=params.FT_NUM_WORKERS
            )
    
    # initialize label counter
    label_count=torch.zeros((params.P_NUM_OUTPUTS), dtype=torch.long, device=params.DEVICE)
    total_count=0

    # go through image data
    with torch.no_grad():
        for i, image in enumerate(train_dl):
            print(f'\rLoad batch {i+1} of {len(train_dl)}.{params.EOL_SPACE}', end='')
            # load region mask
            labeled_region=torch.flatten(image['EO_label'].cpu())
            # total number of pixels
            total_count+= params.P_NUM_PIXELS
            # individual label count
            label_count += torch.bincount(labeled_region, minlength=params.P_NUM_OUTPUTS)
    
    raw_counts=label_count.float().tolist()
    # convert to float
    class_weights={key: float(value) for key, value in params.P_LABELS.items()}
    
    # inverse-frequency weights
    for label_count, label in zip(raw_counts, class_weights.keys()):
        # inverse frequency weighting
        class_weights[label]=(1.0/label_count) if label_count != 0.0 else 0.0
        # print weights to console output
        print(f'Label: {label}, weight: {class_weights[label]}')

    class_weights= np.array(list(class_weights.values()))
    # also returns weights as tensor
    return torch.tensor(class_weights, dtype=torch.float32).to(params.DEVICE)

def main(): 
    '''
    Main function to be called for the calculation of inverse-frequency
    class weights.
    '''

    # calculate class weights
    init_class_weights_pastis()

#main()