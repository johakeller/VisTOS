'''
Defines central parameters for the entire program. Steers general parameters, 
pretraining-dataset-specific parameters, masking-specific paramters, 
model-specific parameters, general fine-tuning hyperparameters, PASTIS-R 
dataset-specific parameters, and MTCC dataset-specific parameters.
'''

import os

import numpy as np
import torch

####################################################################### GENERAL ################################################################################################

# environment used: ['colab', 'cluster',''cluster_portia','local', 'hpc']
environment_= 'hpc'
# general paths
OUTPUT = './output' # output folder
# define cache directory
if environment_ == 'colab':
    CACHE='/content/drive/MyDrive/model_cache' # Colab
else: 
    CACHE = os.path.join (OUTPUT, 'cache/') # cache for model checkpoints
# convert rad to meter: 1 rad equals 11320 meters
FACTOR_METERS_PER_DEG=111320
# end of line spaces in print statements
EOL_SPACE=" "*40
# define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# possible sizes of visual fields
VIS_FIELDS=[1,3,5,7]
# training modes
MODES=('pretrain', 'finetune', 'eval')
# lat lon crs
LATLON_CRS='EPSG:4326'

####################################################################### PRE-TRAINING DATASET ################################################################################################

# urls of pretraining dataset
if environment_ == 'colab':
    TRAIN_URL='/content/gcs/S1_S2_ERA5_SRTM_2020_2021_DynamicWorldMonthly2020_2021_tars/dw_144_shard_{8..39}.tar' # Colab
    VAL_URL='/content/gcs/S1_S2_ERA5_SRTM_2020_2021_DynamicWorldMonthly2020_2021_tars/dw_144_shard_{0..7}.tar' # Colab
else:
    TRAIN_URL='gs://presto-assets2/S1_S2_ERA5_SRTM_2020_2021_DynamicWorldMonthly2020_2021_tars/dw_144_shard_{8..39}.tar'
    VAL_URL='gs://presto-assets2/S1_S2_ERA5_SRTM_2020_2021_DynamicWorldMonthly2020_2021_tars/dw_144_shard_{0..7}.tar'
# set training dataset length in samples
TRAIN_DATA_LENGTH= 42 *1440 # num shards * assumed number of samples per shard 
# set validation dataset length in samples
VAL_DATA_LENGTH=8*1440
IMG_WIDTH=25 # side length in pixels of image
TIME_STEPS = 12 # length of the input data sample timeseries in months
NUM_PIXELS= 625 # number of pixels per image

CHECKPOINT=True # load cached model if available
BATCH_SIZE= 5000 # mini-batch size in samples (pixel time series) only mutliples of 625
# calculate training dataloader length in mini-batches
TRAIN_DL_LENGTH=int(TRAIN_DATA_LENGTH*NUM_PIXELS/BATCH_SIZE)
# calculate validation dataloader length in mini-batches
VAL_DL_LENGTH=int(VAL_DATA_LENGTH*NUM_PIXELS/BATCH_SIZE)
EPOCHS=10 # pretraining epochs
MAX_LR=1e-03 # maximum learning rate (cosine-annealing lr scheduler)
MIN_LR=0.0  # minimum learning rate (cosine-annealing lr scheduler)
DYNAMIC_WORLD_LOSS_WEIGHT=2 # weight for loss calculation
WEIGHT_DECAY=5e-02 # weight decay parameter
WARMUP_EPOCHS=2 # warmup for cosine-annealing lr scheduler

# Original channel indices, before removal of band B9
BAND_IDX={
    'VV': 0, 
    'VH' : 1,
    "B2":2,
    "B3":3,
    "B4":4,
    "B5":5,
    "B6":6,
    "B7":7,
    "B8":8,
    "B8A":9, 
    'B9':10, # will not be used (60 m resolution)
    "B11":11,
    "B12":12,
    "temperature_2m":13,
    "total_precipitation":14,
    'elevation':15,
    'slope':16,
    'NDVI':17
    }

# normalization: additive term
_bands_shift_row = np.array([float(25.0), 
                          float(25.0), 
                          float(0.0), 
                          float(0.0), 
                          float(0.0), 
                          float(0.0), 
                          float(0.0), 
                          float(0.0),
                          float(0.0),
                          float(0.0),
                          float(0.0),
                          float(0.0),
                          float(0.0),
                          float(-272.15), 
                          float(0.0),
                          float(0.0),
                          float(0.0),
                          float(0.0)])
# value to add element-wise to each band
BANDS_SHIFT =np.tile(_bands_shift_row,(24,1))

# normalization: division term
_bands_divide_row = np.array([float(25.0), 
                          float(25.0), 
                          float(1e4), 
                          float(1e4), 
                          float(1e4), 
                          float(1e4), 
                          float(1e4), 
                          float(1e4),
                          float(1e4),
                          float(1e4),
                          float(1e4),
                          float(1e4),
                          float(1e4),
                          float(35.0), 
                          float(0.03),
                          float(2000.0),
                          float(50.0),
                          float(1.0)])
# value to divide by element-wise
BANDS_DIV =np.tile(_bands_divide_row,(24,1))

####################################################################### MASKING ################################################################################################

MASKING_STRATEGIES=('random','channel_groups', 'contiguous_timesteps', 'timesteps') # available masking strategies
MASKING_RATIO=0.75 # how much of the input is masked (in tokens -> one timestep, one channel group)

####################################################################### MODEL ################################################################################################

NUM_HEADS = 8 # number of attention heads per multi-head attention
DEPTH=2 # number of multi-head attention layers per module
DROPOUT=0.1 # dropout ratio (global)
BIAS=0.0 # global bias term
MLP_RATIO=4 # multilayer perceptron ratio for encoder (factor for hidden dimension)
DECODER_MLP_RATIO=2 # multilayer perceptron ratio for decoder (factor for hidden dimension)
LAYER_NORM=False # use layer norm
LAYER_SCALE= None #1e-5 # defines learnable scaling factor for residual connection
QKV_BIAS=True # bias for query, key and value in attention-head
EMBEDDING_SIZE = 128 # one month embedded as vector of length 128
MAX_SEQ_LEN =24 # 24 months max length of time dimension
CHANNEL_EMBED_RATIO =0.25 # proportion of channel embedding in added positional embedding (<=1.0)
MONTHS_EMBED_RATIO =0.25 # proportion of month embedding in added positional embedding (<=1.0)
DW_CLASSES = 9 # number of dynamic world classes
# comprises properties and indices of channel groups
CHANNEL_GROUPS={
    'S1': { # name of the channel group
        'id':0, # index of the channel group
        'length':2, # number of bands in the channel group
        'idx': (0,1) # column index of the bands in the raw numpy input
        },
    'S2_RGB':{
        'id':1,
        'length':3,
        'idx': (2,3,4)
        },
    'S2_Red_Edge':{
        'id':2,
        'length':3,
        'idx': (5,6,7)
        },
    'S2_NIR_10':{
        'id':3,
        'length':1,
        'idx': (8,)
        },
    'S2_NIR_20':{
        'id':4,
        'length':1,
        'idx': (9,)
        },
    'S2_SWIR':{
        'id':5,
        'length':2,
        'idx': (10,11)
        },
    'NDVI':{
        'id':6,
        'length':1,
        'idx': (16,)
        },
    'ERA5':{
        'id':7,
        'length':2,
        'idx': (12,13)
        },
    'SRTM':{
        'id':8,
        'length':2,
        'idx': (14,15)
        },
}

####################################################################### FINE-TUNING: GENERAL HYPERPARAMETERS ################################################################################################

FT_NUM_TRAIN_SAMPLES= None # length of the dataloader dataset in samples (pixels time series)
FT_NUM_VAL_SAMPLES= None # length of the dataloader dataset in samples (pixels time series)
FT_NUM_TEST_SAMPLES=None # length of the dataloader dataset in samples (pixels time series)
FT_WARMUP=5 # warmup epochs for cosine-annealing
FT_MIN_LR= 1e-6 # minimum learning rate
FT_PATIENCE=3 # patience parameter for early stopping
FT_NUM_WORKERS=0 # number of workers for dataloader
FT_DROPOUT=0.1 # global dropout rate
FT_CHECKPOINT=True # load cached model if available
FT_THRESHOLDS=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # 10 thresholds for ROC-AUC
# parameters to exclude from parameter count for Presto
FT_EXCLUDE_PARAMS_PRESTO=['unfold','block_2', 'pix_row_embed', 'pix_col_embed', 'vis_field_rows', 'vis_field_cols' ] # don't count visual field part

####################################################################### FINE-TUNING: PASTIS DATASET ################################################################################################

# paths
# paths Colab
if environment_ == 'colab':
    P_PATH='/content/PASTIS-R/'
# paths Cluster: Hex, Hal
elif environment_ == 'cluster':
    P_PATH='/shared/datasets/PASTIS-R/'
# Portia
elif environment_ == 'cluster_portia':
    P_PATH=os.path.expanduser('~/data/PASTIS-R/')
# HPC
elif environment_ =='hpc':
    P_PATH='/scratch/johakeller/datasets/PASTIS-R/'
# default: local path
else:
    P_PATH='/home/johakeller/Documents/Master_Computer_Science/Master_Thesis/Workspace2/data/PASTIS-R/' # local path to PASTIS-R
P_METADATA=os.path.join(P_PATH,'metadata.geojson') # path to PASTIS-R metadata
P_S1A_PATH=os.path.join(P_PATH,'DATA_S1A') # path to PASTIS-R Sentinel-1 Ascending data
P_S1D_PATH=os.path.join(P_PATH,'DATA_S1D') # path to PASTIS-R Sentinel-1 Descending data
P_S2_PATH=os.path.join(P_PATH,'DATA_S2') # path to PASTIS-R Sentinel-2 data

P_IMG_WIDTH=128 # width of a PASTIS-R image
P_NUM_PIXELS=16384 # number of pixels per image (128^2)
P_MAX_SEQ_LEN=12 #12 months maximum sequence length for PASTIS-R
P_BATCH_SIZE=P_IMG_WIDTH*8 # PASTIS-R number of samples (pixels time series) -> should be more than vis_field_size*FT_IMG_WITDH (otherwise too much padding)
P_MAX_EPOCHS= 10 # number of fine-tuning epochs
P_MAX_LR=1e-4 # maximum lerning rate for PASTIS-R
P_MIN_LR=1e-6
P_WEIGHT_DECAY=0.01 # weight decay term for PASTIS-R 
P_DROPOUT=0.1

# params for multi-class segmentation
P_TVERSKY_ALPHA=0.6 # penalization for false positives (FTL)
P_TVERSKY_BETA=0.4 # penalization for false negatives (FTL)
P_TVERSKY_GAMMA=1.5 # focus on rare classes (FTL)
P_TVERSKY_CLASSES=list(range(0,19)) # classes to not ignore for FTl, ignores only 19 -> index 20
P_LAMBDA_1=0.7 # Tversky ratio in combined loss
P_LAMBDA_2=1-P_LAMBDA_1# cross-enrtopy ratio in combined loss

# dataset pipeline
# labels to int
P_LABELS={
    'Background':0,
    'Meadow': 1,
    'Soft winter wheat': 2,
    'Corn' : 3,
    'Winter barley' : 4,
    'Winter rapeseed': 5, 
    'Spring barley': 6,
    'Sunflower':7,
    'Grapevine':8,
    'Beet':9,
    'Winter triticale': 10,
    'Winter durum wheat':11,
    'Fruits, vegetables, flowers':12,
    'Potatoes':13,
    'Leguminous fodder':14,
    'Soybeans':15,
    'Orchad': 16,
    'Mixed cereal':17,
    'Sorghum':18,
    'Void label':19,
}
# to address label name via value
P_LABELS_INV={
    value:key for key, value in P_LABELS.items()
}
P_NUM_OUTPUTS=len(P_LABELS) # number of classes

# inverse frequency weights
p_weights_= {
    'Background':1.050e-07,
    'Meadow': 2.306e-07,
    'Soft winter wheat': 6.279e-07,
    'Corn' : 4.667e-07,
    'Winter barley' : 2.102e-06,
    'Winter rapeseed': 2.391e-06, 
    'Spring barley': 6.867e-06,
    'Sunflower':4.489e-06,
    'Grapevine':1.424e-06,
    'Beet':5.370e-06,
    'Winter triticale': 5.501e-06,
    'Winter durum wheat':3.828e-06,
    'Fruits, vegetables, flowers':4.336e-06,
    'Potatoes':1.444e-05,
    'Leguminous fodder':2.675e-06,
    'Soybeans':3.572e-06,
    'Orchad': 3.834e-06,
    'Mixed cereal':8.716e-06,
    'Sorghum':1.224e-05,
    'Void label':0, # ignore this class with CE ignore index
}
P_WEIGHTS=torch.tensor(list(p_weights_.values())) 
P_DELTA=0.5 # weight dampen parameter
# to represent labels as RGB-colors
P_CLASS_COLORS=[
    (0, 0, 0),
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (1.0, 0.7333333333333333, 0.47058823529411764),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (1.0, 0.596078431372549, 0.5882352941176471),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    (1, 1, 1)
]

####################################################################### FINE-TUNING: MTC DATASET ################################################################################################

# paths
# paths Colab
if environment_ == 'colab':
    MTCC_PATH='/content/multi-temporal-crop-classification/'
# paths Cluster: Hex, Hal
elif environment_ == 'cluster':
    MTCC_PATH='/shared/datasets/multi-temporal-crop-classification/'
# Portia
elif environment_ == 'cluster_portia':
    MTCC_PATH=os.path.expanduser('~/multi-temporal-crop-classification/')
# HPC
elif environment_ =='hpc':
    MTCC_PATH='/scratch/johakeller/datasets/multi-temporal-crop-classification/'
# default: local path
else:
    MTCC_PATH='/home/johakeller/Documents/Master_Computer_Science/Master_Thesis/Workspace2/data/multi-temporal-crop-classification/'

MTCC_IMG_WIDTH=224 # length of square side of image
MTCC_NUM_PIXELS=MTCC_IMG_WIDTH**2 # number of pixels per image
MTCC_MAX_SEQ_LEN=12
MTCC_TIME_STEPS=3
MTCC_NUMBER_CHANNELS=6
MTCC_BATCH_SIZE= 7168 #1792 # number of samples (pixels time series) -> should be more than vis_field_size*FT_IMG_WITDH (otherwise too much padding)
MTCC_MAX_EPOCHS= 5 # number of fine-tuning epochs
MTCC_MAX_LR=3e-5 # maximum lerning rate 
MTCC_MIN_LR=1e-6
MTCC_WEIGHT_DECAY=0.04 # weight decay term 
MTCC_DROPOUT=0.25 # global dropout rate

# coordinates are not available, calculate random coordinates from range of Brazilian Amazon [(min lat, max lat),(min long, max lon)]
MTC_COORD_RANGE=[(-15, 5),(-75,-45)]

# number of classes
MTCC_NUM_OUTPUTS=14

MTCC_TVERSKY_ALPHA=0.3 # penalization for false positives (FTL)
MTCC_TVERSKY_BETA=0.7 # penalization for false negatives (FTL)
MTCC_TVERSKY_GAMMA=1.33 # focus on rare classes (FTL)
MTCC_LAMBDA_1=0.7 # FTL ratio in combined loss
MTCC_LAMBDA_2=1-MTCC_LAMBDA_1 # cross-entropy ratio in combined loss
MTCC_CE_LABEL_SMOOTHING=0.1 # label smoothing term for cross-entropy

MTCC_CHANNEL_GROUPS={
    'S2_RGB':[0,1,2],
    'S2_NIR_20':[3],
    'S2_SWIR':[4,5]
}
# normalization factor
MTCC_NORM=1.0/10000.0

MTCC_LABELS={
    'No Data':0,
    'Natural Vegetation': 1,
    'Forest': 2,
    'Corn' : 3,
    'Soybeans' : 4,
    'Wetlands': 5, 
    'Developed/Barren': 6,
    'Open Water':7,
    'Winter Wheat':8,
    'Alfalfa':9,
    'Fallow/Idle Cropland': 10,
    'Cotton':11,
    'Sorghum':12,
    'Other':13 
}
# to address label name via value
MTCC_LABELS_INV={
    value:key for key, value in MTCC_LABELS.items()
}

# inverse frequency weights (roughly estimated)
mtcc_weights_= {
    'No Data':0.0,
    'Natural Vegetation': 1.0/0.2,
    'Forest': 1.0/0.12,
    'Corn' : 1.0/0.14,
    'Soybeans' : 1.0/0.12,
    'Wetlands': 1.0/0.08, 
    'Developed/Barren': 1.0/0.079,
    'Open Water':1.0/0.025,
    'Winter Wheat':1.0/0.04,
    'Alfalfa':1.0/0.035,
    'Fallow/Idle Cropland': 1.0/0.035,
    'Cotton':1.0/0.025,
    'Sorghum':1.0/0.02,
    'Other':1.0/0.07 
}
MTCC_CLASSES=list(range(1,len(mtcc_weights_)))

# norm to sum: number of classes
mtcc_weights_=torch.tensor(list(mtcc_weights_.values()))
MTCC_WEIGHTS=mtcc_weights_*(len(mtcc_weights_)/mtcc_weights_.sum())

# to represent labels as RGB-colors
MTCC_CLASS_COLORS=[
    (0, 0, 0),
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (1.0, 0.7333333333333333, 0.47058823529411764),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (1.0, 0.596078431372549, 0.5882352941176471),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
]

