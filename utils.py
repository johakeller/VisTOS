'''
Auxiliary module for visualizations of images and predictions of all used datasets,
as well as initialization of logger and creation of output and cache directories.
'''

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
from sklearn.metrics import RocCurveDisplay

import params


def init_output(output_dir=params.OUTPUT, cache=params.CACHE):
    '''
    Creates output directory with cache, which will house cached models,
    if it does not exist yet.
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(cache):
        os.makedirs(cache)
    
def init_logger(output_dir=params.OUTPUT, name ='pretraining'):
    '''
    Initializes logging and returns logger object. 
    Credits: Tseng et al., 2023:
    https://github.com/nasaharvest/presto/blob/main/presto/utils.py#L48
    '''

    logger=logging.getLogger(name)
    # prevent duplicate logs
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(message)s"
    )

    logger.setLevel(logging.INFO)

    # specify saving path
    path=os.path.join(output_dir, name)
    fh = logging.FileHandler(path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def visualize_images_pretraining(eo_data):
    '''
    Visualizes every fourth month of an input-image time series from the pretraining
    dataset in the form of: VV band from Sentinel-1, RGB channels from Sentinel-2, 
    NDVI from Sentinel-2. 
    '''

    # get number of available time steps from EO data
    _, num_ts, num_chan=eo_data.shape
    img_width=params.IMG_WIDTH
    # reconstruct 2D images from EO data
    eo_data_2d= eo_data.reshape(img_width, img_width,num_ts, num_chan)
    
    font_size=24
    fig, axs=plt.subplots(nrows=3, ncols=6,figsize=(22,8))

    # iterate through timesteps (every 4th month) of EO data
    for ts_id, timestep in enumerate(list(range(0,num_ts,4))):
        # assign months to columns
        month = timestep+1
        axs[0,ts_id].set_title(f'Month {month}', fontsize= font_size)

        # display VV band (Sentinel-1)
        sar_data=eo_data_2d[:,:,timestep,0]
        axs[0,ts_id].imshow(sar_data, cmap='magma')
        axs[0, 0].set_ylabel('Sentinel-1\nVV', fontsize=font_size, rotation=90)

        # display RGB (Sentinel-2)
        rgb_data=eo_data_2d[:,:,timestep,[4,3,2]]
        axs[1,ts_id].imshow(rgb_data)
        axs[1, 0].set_ylabel('Sentinel-2\nRGB', fontsize=font_size, rotation=90)

        # display NDVI (derived from Sentinel-2)
        ndvi_data=eo_data_2d[:,:,timestep,16]
        axs[2,ts_id].imshow(ndvi_data, cmap='RdYlGn')
        axs[2, 0].set_ylabel('NDVI', fontsize=font_size, rotation=90)

        # set axs and tics invisible
        for row in range (3):
            axs[row, ts_id].tick_params(
                axis='both',           
                which='both',          
                left=False,            
                bottom=False,          
                labelleft=False,       
                labelbottom=False      
            )
            for spine in axs[row, ts_id].spines.values():
                spine.set_visible(False)

    fig.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    plt.show()

def visualize_prediction_pastis(
    eo_data: torch.Tensor, 
    preds:torch.Tensor, 
    label:torch.Tensor, 
    title: str, 
    timestep: int =0, 
    image_path: str =params.OUTPUT,
    ):
    '''
    Visualizes model predictions on the PASTIS-R dataset as an image and saves the result. 
    Plots an output in the form: VV-band from Sentinel-1 (for passed time step), RGB-image 
    from Sentinel-2 (for passed time step), label map, model prediction from Sentinel-2. 
    '''

    # encode predicted class as integer
    pred_classes = torch.argmax(preds, dim=1)
    
    # slice the image from flattened pixel array
    num_chan=eo_data.shape[-1]
    # reshape to 2D
    img_width=params.P_IMG_WIDTH
    label_img=label.reshape(img_width,img_width)
    pred_img=pred_classes.reshape(img_width,img_width)
    eo_img=eo_data[:,timestep].reshape(img_width,img_width, num_chan)

    # to numpy array
    pred_img=pred_img.detach().cpu().numpy() 
    label_img = label_img.cpu().numpy() 
    eo_img=eo_img.cpu().numpy()
    
    _, axs =plt.subplots(1,4, figsize=(15,5))

    # SAR, VV polarization
    eo_sar=eo_img[:,:,0]
    axs[0].imshow(eo_sar, cmap='magma')
    axs[0].set_title('SAR VV')
    axs[0].axis('off')

    # RGB
    eo_rgb = eo_img[:,:,[4,2,3]]
    # show rgb image
    axs[1].imshow(eo_rgb)
    axs[1].set_title('RGB')
    axs[1].axis('off')

    # label
    label_cmap=ListedColormap(params.P_CLASS_COLORS)
    # multi-class
    axs[2].imshow(label_img, cmap=label_cmap, vmin=0, vmax=19)
    axs[2].set_title('Label')
    axs[2].axis('off')

    # prediction
    axs[3].imshow(pred_img, cmap=label_cmap, vmin=0, vmax=19)
    axs[3].set_title('Prediction')
    axs[3].axis('off')

    # save to image path
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    out_path=os.path.join(image_path, f'prediction_{title}.png')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.3)
    plt.close()

def visualize_prediction_mtcc(
    eo_data: torch.Tensor, 
    preds:torch.Tensor, 
    label:torch.Tensor, 
    title: str, 
    timestep: int =0, 
    image_path: str =params.OUTPUT,
    ):
    '''
    Visualizes model predictions on the MTCC dataset as an image and saves the result. 
    Plots an output in the form: RGB-image from Sentinel-2 (for passed time step), 
    NDVI-band from Sentinel-1 (for passed time step), label map, model prediction. 
    '''

    # encode predicted class as integer
    pred_classes = torch.argmax(preds, dim=1)
    
    # slice the image from flattened pixel array
    num_chan=eo_data.shape[-1]
    # reshape to 2D
    img_width=params.MTCC_IMG_WIDTH
    label_img=label.reshape(img_width,img_width)
    pred_img=pred_classes.reshape(img_width,img_width)

    # search for non-empty timestep
    eo_rgb_tmp=eo_data[:,timestep]
    while eo_data[:,timestep,[4,2,3]].sum()==0.0 and timestep<(params.MTCC_MAX_SEQ_LEN-1):
        timestep +=1
        eo_rgb_tmp=eo_data[:,timestep]
    eo_img=eo_rgb_tmp.reshape(img_width,img_width, num_chan)

    # to numpy array
    pred_img=pred_img.detach().cpu().numpy() 
    label_img = label_img.cpu().numpy() 
    eo_img=eo_img.cpu().numpy()
    
    _, axs =plt.subplots(1,4, figsize=(15,5))

    # RGB
    eo_rgb = eo_img[:,:,[4,2,3]]
    # show rgb image
    axs[0].imshow(eo_rgb)
    axs[0].set_title('RGB')
    axs[0].axis('off')

    # NDVI
    eo_sar=eo_img[:,:,16]
    axs[1].imshow(eo_sar, cmap='RdYlGn')
    axs[1].set_title('NDVI')
    axs[1].axis('off')

    # label
    label_cmap=ListedColormap(params.MTCC_CLASS_COLORS)
    # multi-class
    axs[2].imshow(label_img, cmap=label_cmap, vmin=0, vmax=13, interpolation='nearest')
    axs[2].set_title('Label')
    axs[2].axis('off')

    # prediction
    axs[3].imshow(pred_img, cmap=label_cmap, vmin=0, vmax=13, interpolation='nearest')
    axs[3].set_title('Prediction')
    axs[3].axis('off')

    # save to image path
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    out_path=os.path.join(image_path, f'prediction_{title}.png')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.3)
    plt.close()

def roc_auc_curve_mtcc(prediction: NDArray[Any], label: NDArray[Any], image_path: str =params.OUTPUT,):
    '''
    Plots multi-class AUC-ROC curves (for each class individually) for MTCC 
    dataset for raw predictions and labels in shape (batch size, number of 
    channels) and saves it to passed directory path.
    '''

    fig, ax = plt.subplots(figsize=(6, 6))
    # only plot three rare and three frequent classes: 
    # (Natural Vegetation=1, Corn=3, Soybeans=4, Cotton=11, Open Water=7, Sorghum=12)
    classes_display=[1,3,4,11,7,12]
    # Get corresponding colors
    label_colors=[params.MTCC_CLASS_COLORS[class_id] for class_id in classes_display] 

    # iterate through classes and class-colors
    for class_id, color in zip(classes_display, label_colors):
        # get label for class
        label_class = (label==class_id).astype(int)
        # prediction for class
        predicted_class=prediction[:,class_id]
        RocCurveDisplay.from_predictions(
            label_class, 
            predicted_class,
            name=params.MTCC_LABELS_INV[class_id],
            color=color,
            ax=ax,
            # chance level at last
            plot_chance_level=(class_id==classes_display[-1]),
        )
    ax.set_xlabel('False positive rate', fontsize=14)
    ax.set_ylabel('True positive rate',fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='lower right', fontsize=14)

    # save figure
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    out_path=os.path.join(image_path, 'MTCC_ROC_AUC.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def roc_auc_curve_pastis(prediction: NDArray[Any], label: NDArray[Any], image_path: str =params.OUTPUT,):
    '''
    Plots multi-class AUC-ROC curves (for three rare classes and for three
    frequent classes) for PASTIS-R dataset for raw predictions and 
    labels in shape (batch size, number of channels) and saves it to 
    passed directory path.
    '''

    fig, ax = plt.subplots(figsize=(6, 6))
    # only plot three rare and three frequent classes: 
    # (background=0, meadow=1, corn=3, potatoes=13, sorghum=18, spring barley=6)
    classes_display=[0,1,3,6,13,18]
    # Get corresponding colors
    label_colors=[params.P_CLASS_COLORS[class_id] for class_id in classes_display] 
    
    # iterate through classes and class-colors
    for class_id, color in zip(classes_display, label_colors):
        # label for class
        label_class = (label==class_id).astype(int)
        # prediction for class
        predicted_class=prediction[:,class_id]
        RocCurveDisplay.from_predictions(
            label_class, 
            predicted_class,
            name=params.P_LABELS_INV [class_id],
            color=color,
            ax=ax,
            # chance level at last
            plot_chance_level=(class_id==classes_display[-1]),
        )
    ax.set_xlabel('False positive rate', fontsize=14)
    ax.set_ylabel('True positive rate',fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='lower right', fontsize=14)
    
    # save figure
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    out_path=os.path.join(image_path, 'PASTIS-R_ROC_AUC.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)