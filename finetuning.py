'''
Module to initiate fine-tuning. The entry-point is init_finetuning() of the class FineTuning.
Standard fine-tuning includes training and subsequent evaluation, but an already trained
model can also directly be evaluated.
'''

import logging
import math
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from segmentation_models_pytorch.losses import TverskyLoss
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import params
import utils
from dataset import multitempcrop_dataset, pastis_r_dataset
from model import vistos_att_model

# set scaler for mixed precision training 
scaler = torch.GradScaler(enabled=torch.cuda.is_available())

class FineTuning:
    '''
    This class conducts fine-tuning with the entry point init_finetuning(). Fine-tuning
    includes training, validation and testing of a pretrained model. Uses early 
    stopping for training. During testing several evaluation metrics are applied 
    and saved to output file. Also visualizations of predictions and image 
    samples are saved during testing using the utils module. Can also run 
    evaluation directly without preceding training. Trained models are saved 
    in the model cache in the output directory and automatically loaded from there. 
    '''

    def __init__(self, vis_field_size, dataset):
        self.dataset = 'PASTIS-R' if dataset == 'pastis' else 'MTCC'
        self.logger = logging.getLogger('default')
        self.vis_field_size = vis_field_size
        self.model_type = 'att'
        self.image_path= ''
        self.warmup_epochs = params.FT_WARMUP
        self.min_learning_rate=params.FT_MIN_LR

    def init_finetuning(
        self, 
        pretrained_model, 
        checkpointing=True, 
        eval_mode=False, 
        model_type='att',
        model_class=vistos_att_model,
        vis_field_size=params.VIS_FIELDS[0],
    ):
        '''
        Start of the finetuning logic for fine-tuning of semantic segmentation. Initializes the 
        logger, tries to load a model from cache first, if cache is empty, starts fine-tuning from
        scratch (loads pretrained model from output directory). Initializes dataset-objects, 
        dataloaders and starts fine-tuning followed by evaluation. If argument eval_mode passed, 
        directly jumps into evaluation. Method has two separate branches: one for PASTIS-R dataset
        and the other for the BraDD-S1TS dataset. The method can handle two types of models, the VisTOS VF
        model with attention-based spatial enoding and the VisTOS CVF model with convolutional 
        spatial encoding. The type is passed via model_type argument. Cache-loading behavior is 
        defined via checkpointing argument.
        '''
        # init logger
        self.logger = utils.init_logger(name=f'finetuning_{self.dataset}_{model_type}_vf{self.vis_field_size}')
        # set start epoch, start iteration, previous loss to 0
        epoch = 0
        iteration = 0
        loss = 0
        # set model type and image path
        self.model_type = model_type
        self.image_path = os.path.join(params.OUTPUT, f'{model_type}_{self.vis_field_size}_{self.dataset}_images')

        # PASTIS-R branch
        if self.dataset == 'PASTIS-R':
            # create pretrained Seq2Seq model (uses pretrained model from output or preferably cache if there is any)
            pretrained_model=model_class.VistosTimeSeriesSeq2Seq.load_pretrained(vis_field_size=vis_field_size, dropout=params.P_DROPOUT).to(params.DEVICE)
        
            # construct the fine-tunig model from the pre-trained model
            model = pretrained_model.construct_finetuning_model(
                num_outputs=params.P_NUM_OUTPUTS,
                vis_field_size=self.vis_field_size,
                img_width=params.P_IMG_WIDTH,
            ).to(params.DEVICE)
            # training dataset
            train_ds = pastis_r_dataset.PastisRDataset(split='train', max_length=params.FT_NUM_TRAIN_SAMPLES, shuffle=True)
            # validation dataset
            val_ds = pastis_r_dataset.PastisRDataset(split='validation', max_length=params.FT_NUM_VAL_SAMPLES, shuffle=True)
            # test dataset
            test_ds = pastis_r_dataset.PastisRDataset(split='test', max_length=params.FT_NUM_TEST_SAMPLES, shuffle=True)
            # training params
            self.total_pixels = params.P_NUM_PIXELS
            batch_size=params.P_BATCH_SIZE
            self.epochs = params.P_MAX_EPOCHS
            self.max_learning_rate = params.P_MAX_LR
            self.min_learning_rate = params.P_MIN_LR
            weight_decay=params.P_WEIGHT_DECAY
            vis_method=utils.visualize_prediction_pastis
            # exclude Void class
            self.label_list = list(range(19))
            # FTL params
            classes=params.P_TVERSKY_CLASSES
            from_logits=True
            alpha=params.P_TVERSKY_ALPHA
            beta=params.P_TVERSKY_BETA
            gamma=params.P_TVERSKY_GAMMA
            eps=1e-4
            mode='multiclass'
            # CE params
            ignore_index=19
            label_smoothing=0.1
            weight=(params.P_WEIGHTS.to(params.DEVICE))**params.P_DELTA
            # combined loss params
            lambda_1=params.P_LAMBDA_1
            lambda_2=params.P_LAMBDA_2

        # MTCC branch
        elif self.dataset == 'MTCC':
            # create pretrained Seq2Seq model (uses pretrained model from output or preferably cache if there is any)
            pretrained_model=model_class.VistosTimeSeriesSeq2Seq.load_pretrained(vis_field_size=vis_field_size, dropout=params.MTCC_DROPOUT).to(params.DEVICE)
            # construct the fine-tunig model from the pretrained model
            model = pretrained_model.construct_finetuning_model(
                num_outputs=params.MTCC_NUM_OUTPUTS,
                vis_field_size=self.vis_field_size,
                img_width=params.MTCC_IMG_WIDTH,
            ).to(params.DEVICE)
            # training dataset
            train_ds = multitempcrop_dataset.MultiTempCropClass(split='train',shuffle=True)
            # validation dataset
            val_ds = multitempcrop_dataset.MultiTempCropClass(split='validation',shuffle=True)
            # test dataset
            test_ds = multitempcrop_dataset.MultiTempCropClass(split='test',shuffle=True)
            # training params
            self.total_pixels = params.MTCC_NUM_PIXELS
            batch_size=params.MTCC_BATCH_SIZE
            self.epochs = params.MTCC_MAX_EPOCHS
            self.max_learning_rate = params.MTCC_MAX_LR
            self.min_learning_rate = params.MTCC_MIN_LR
            weight_decay=params.MTCC_WEIGHT_DECAY
            vis_method=utils.visualize_prediction_mtcc
            # exclude No Data class
            self.label_list = list(range(1,14))
            # CE params
            weight = params.MTCC_WEIGHTS.to(params.DEVICE)
            label_smoothing=params.MTCC_CE_LABEL_SMOOTHING
            # default
            ignore_index=0
            # FTL params
            classes=params.MTCC_CLASSES
            from_logits=True
            alpha=params.MTCC_TVERSKY_ALPHA
            beta=params.MTCC_TVERSKY_BETA
            gamma=params.MTCC_TVERSKY_GAMMA
            eps=1e-4
            mode='multiclass'
            # combined loss params
            lambda_1=params.MTCC_LAMBDA_1
            lambda_2=params.MTCC_LAMBDA_2
        
        # unknown dataset
        else:
            raise ValueError(f'Dataset {self.dataset} unknown.')
        
        # print and log info
        message = (
            f'\rFine-tuning VisTOS {"VF size" if model_type == "att" else "CVF size"} {self.vis_field_size} on {self.dataset}, '
            f'batch size {batch_size}, '
            f'{datetime.now().strftime("%d-%m-%Y %H:%M")}\t'
        )
        print(message)
        self.logger.info(message)

        # Focal Tversky loss for rare classes
        loss_1 = TverskyLoss(
            mode=mode,
            classes=classes,
            from_logits=from_logits,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            eps=eps,
        )
        # CE for foreground/background separation and stability
        loss_2 = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing, 
            ignore_index=ignore_index, 
            reduction='mean'
        )

        # define optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.max_learning_rate,
            weight_decay=weight_decay,
            eps=1e-6,
        )
        # try loading a model from cache if available
        if checkpointing:
            model, optimizer, epoch, iteration, loss = self.load_checkpoint(
                model, 
                optimizer, 
                epoch, 
                iteration, 
                loss
            )
        # dataloaders: training dataloader
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=params.FT_NUM_WORKERS,
            drop_last=True,
        )
        # validation dataloader
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=params.FT_NUM_WORKERS,
            drop_last=True,
        )
        # test dataloader
        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=params.FT_NUM_WORKERS,
            drop_last=True,
        )
        # start fine-tuning procedure if eval_mode is False
        if not eval_mode:
            # run fine-tuning, returns the fine-tuned model
            model = self.finetune(
                model,
                optimizer,
                train_dl,
                val_dl,
                loss_1=loss_1,
                loss_2=loss_2,
                start_epoch=epoch,
                start_iteration=iteration,
                start_loss=loss,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
            )
        # test fine-tuned model (directly accessed with eval_mode)
        results_dict = self.evaluate(model, test_dl, vis_method=vis_method)        
        # log multi-class metrics dictionary content to ouput
        self.logger.info('Test metrics:')
        for metric, result in results_dict.items():
            # don't round dictionaries of metrics or confusion matrix
            if metric == 'confusion_matrix' or isinstance(result, dict):
                self.logger.info('%s: %s', metric, result)
            else:
                self.logger.info('%s: %.3f', metric, result)

    @staticmethod
    def adjust_learning_rate(
        optimizer, 
        epoch, 
        warmup_epochs, 
        total_epochs, 
        max_lr, 
        min_lr,
    ):
        '''
        Decays the learning rate with half-cycle cosine after warmup.
        '''
        if epoch < warmup_epochs:
            # starts at min_lr
            lr = min_lr + ((max_lr - min_lr) * epoch / warmup_epochs)
        else:
            lr = min_lr + (max_lr - min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                )
            )
        for param_group in optimizer.param_groups:
            if 'lr_scale' in param_group:
                # This is only used during finetuning
                param_group['lr'] = lr * param_group['lr_scale']
            else:
                param_group['lr'] = lr
        return lr

    def compound_loss(
        self,
        loss_1,
        loss_2,
        lambda_1,
        lambda_2
    ):
        '''
        Computes weighted compound loss of fine-tuning head:
        Loss_total = lambda_1*Loss_binary +lambda_2*Loss_CE.
        '''

        return (lambda_1 * loss_1) + (lambda_2 * loss_2)

    def load_checkpoint(
        self, 
        model, 
        optimizer, 
        epoch, 
        iteration, 
        loss,
    ):
        '''
        Loads a model from model checkpoint if one is available. Reproduces
        and returns cached model, optimizer, iteration, epoch, and loss 
        calculated so far. 
        '''

        try:
            # try to fetch list of saved model checkpoints: check for model type 
            # (attention/convolution) and correct visual field size
            old_checkpoints = [file for file in os.listdir(params.CACHE) if f'{self.model_type}_model_ft_{self.vis_field_size}' in file]
            # if more than on suitable model found
            if len(old_checkpoints) > 1:
                print(f'\rMore than one checkpoint available, loading last checkpoint: {old_checkpoints[-1]}.{params.EOL_SPACE}',end='')
                # fetch the last saved model
                old_checkpoint = old_checkpoints[-1]
            # only one model in list
            else:
                old_checkpoint = old_checkpoints[0]
            # load the selected model_checkpoint dictionary
            model_checkpoint = torch.load(
                os.path.join(params.CACHE, old_checkpoint),
                weights_only=True,
                map_location=params.DEVICE,
            )
            # update model
            model.load_state_dict(model_checkpoint['model'])
            # update optimizer
            optimizer.load_state_dict(model_checkpoint['optimizer'])
            # update training epoch
            epoch = model_checkpoint['epoch']
            # update training iteration
            iteration = model_checkpoint['iteration']
            # introduced after training of some models
            loss = model_checkpoint.get('loss', 0.0)
        # cache is empty
        except IndexError:
            print(f'\rNo checkpoint found, starting from scratch.{params.EOL_SPACE}',end='')
        # return cached model and training state
        return model.to(params.DEVICE), optimizer, epoch, iteration, loss

    def update_checkpoint(
        self, 
        model, 
        optimizer, 
        epoch, 
        iteration, 
        loss,
    ):
        '''
        Saves current state of model and optimizer to checkpoint, removes old
        checkpoints.
        '''
        # current state of training to dictionary: epoch, iteration, model, optimizer, loss
        model_dict = {
            'epoch': epoch,
            'iteration': iteration,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
        }

        # remove old checkpoint with same visual field size and same model type
        old_checkpoints = [check_point for check_point in os.listdir(params.CACHE) if f'{self.model_type}_model_ft_{self.vis_field_size}' in check_point]
        for file in old_checkpoints:
            os.remove(os.path.join(params.CACHE, file))

        # save new checkpoint
        model_path=os.path.join(params.CACHE, f'{self.model_type}_model_ft_{self.vis_field_size}_{datetime.now().strftime("%m_%d_%H")}.pth')
        torch.save(model_dict,model_path)

    def count_params(self, model):
        '''
        Method to count the number of trainable parameters in the model.
        '''

        # define params to exclude from count
        # Presto: exclude attention-based visual field form counting
        if self.vis_field_size == 1:
            exclude_list = params.FT_EXCLUDE_PARAMS_PRESTO
        else:
            exclude_list = []

        params_sum = 0
        # iterate through model parameters
        for param_name, param in model.named_parameters():
            # skip params without gradient
            if not param.requires_grad:
                continue
            # skip params from exclude list
            if any(exclude_param in param_name for exclude_param in exclude_list):
                continue
            # if not continued, count param
            params_sum += param.numel()
        # print output only 
        print(f'\nTrainable params: {params_sum}.')

    @staticmethod
    def class_confidence(labels_true, prob_pred, labels, offset=0):
        '''
        Calculates average confidences for positive instances of a label and returns a dictionary:
        label (int): confidence (float).
        '''

        conf_dict = {}

        for label in labels:
            # mask for true instances for class label
            true_mask = labels_true == label
            # mean over label true instances
            conf_dict[label] = prob_pred[true_mask, label-offset].mean()
        return conf_dict

    def compute_metrics(self, raw_pred, label):
        '''
        Computes various metrics given prediction logit tensors and label tensors and returns 
        them in a dictionary for further averaging. Also calls the AUC-ROC curve function from
        module utils, which visualizes ROC-curves individually for PASTIS-R and BraDD-S1TS.
        '''

        # label to numpy
        label_np = label.cpu().numpy().ravel()
        # ignore void label in PASTIS-R
        if self.dataset == 'PASTIS-R':
            # mask class 19 instances
            void_mask = label_np != 19
            label_np = label_np[void_mask]
            # delete column label 19 from raw predictions
            raw_pred = raw_pred[:, :-1]
        # ignore 'No Data' label in MTCC
        else:
            # mask class 0 instances
            void_mask = label_np != 0
            label_np = label_np[void_mask]
            # delete column label 19 from raw predictions
            raw_pred = raw_pred[:,1 :]

        # remove possible NaNs
        raw_pred = torch.nan_to_num(raw_pred, nan=0.0, posinf=1000, neginf=-1000)

        # logit predictions to probabilities
        prob_pred = torch.softmax(raw_pred, dim=1).cpu().numpy()[void_mask]
        # class predictions, for MTCC compensate for off by 1
        prediction = np.argmax(raw_pred.cpu().numpy(), axis=1).ravel()[void_mask] if self.dataset == 'PASTIS-R' else  np.argmax(raw_pred.cpu().numpy(), axis=1).ravel()[void_mask] +1

        # is off by one for MTCC
        

        # get values for ROC-AUC visualization
        if self.dataset == 'PASTIS-R':
            utils.roc_auc_curve_pastis(prob_pred, label_np, image_path=self.image_path)
        else:
            utils.roc_auc_curve_mtcc(prob_pred, label_np, image_path=self.image_path)

        # metrics: average macro -> simple average for imbalanced datasets
        metrics = {
            'precision': precision_score(
                label_np, prediction, labels=self.label_list, average='macro', zero_division=0
            ),
            'f1_score': f1_score(
                label_np, prediction, labels=self.label_list, average='macro', zero_division=0
            ),
            'recall': recall_score(
                label_np, prediction, labels=self.label_list, average='macro', zero_division=0
            ),
            'accuracy': accuracy_score(label_np, prediction),
            'iou': jaccard_score(
                label_np, prediction, labels=self.label_list, average='macro', zero_division=0
            ),
            'roc_auc':roc_auc_score(
                label_np, prob_pred, labels=self.label_list, average='macro', multi_class='ovo'
            ),
            # different indexing for MTCC
            'confidence': self.class_confidence(label_np, prob_pred, labels=self.label_list) if self.dataset=='PASTIS-R' else self.class_confidence(label_np, prob_pred, labels=self.label_list, offset=1),
            # return confusion matrix as list
            'confusion_matrix': confusion_matrix(
                label_np, prediction, labels=self.label_list
            ).tolist(),
        }
        return metrics

    def finetune(
        self,
        model,
        optimizer,
        train_dl,
        val_dl,
        loss_1,
        loss_2,
        start_epoch=0,
        start_iteration=0,
        start_loss=0,
        lambda_1=params.P_LAMBDA_1,
        lambda_2=params.P_LAMBDA_2,
        count_params=True,
    ):
        '''
        Runs fine-tuning training and validation procedure with passed dataset in a hold-out validation
        scheme. Applies early stopping and returns best-performing model during validation. Performes 
        mixed-precision training and saves model checkpoints periodically during training. Skips to 
        epoch and iteration if caches model is passed. 
        '''

        # count trainble params of model
        if count_params:
            self.count_params(model)
        # early stoppping variables
        patience = params.FT_PATIENCE
        epochs_since_improvement = 0
        best_loss = None
        best_model_dict = None

        # general training variables
        dataloader_len = len(train_dl)
        train_loss = []
        val_loss = []

        # iterate through epochs (starting at start epoch)
        for epoch in (pbar := tqdm(range(start_epoch, self.epochs), desc='Fine-tuning epoch')):
            # TRAINING
            model.train()
            # init with the saved loss for cached model, with 0 for every new epoch
            epoch_train_loss = start_loss if start_epoch == epoch else 0
            # iteration counts from 0, num_batches from 1
            num_batches = start_iteration + 1 if start_iteration != 0 else 0

            # iterate through the mini-batches
            for iteration, input_dict in enumerate(tqdm(train_dl, desc='Training', leave=True, dynamic_ncols=True)):
                # skip until saved epoch and iteration:
                if epoch == start_epoch and iteration < start_iteration:
                    print(f'\rLoaded cached model: skip epoch {epoch+1} iteration {iteration}{params.EOL_SPACE}',end='')
                    continue

                # extract labels from dictionary
                label = input_dict['EO_label'].long()
                optimizer.zero_grad()
                # learning rate scheduling via cosine annealing
                _ = self.adjust_learning_rate(
                    optimizer=optimizer,
                    epoch=iteration / dataloader_len + epoch,
                    warmup_epochs=self.warmup_epochs,
                    total_epochs=self.epochs,
                    max_lr=self.max_learning_rate,
                    min_lr=self.min_learning_rate,
                )
                # forward pass with mixed precision training (if cuda available)
                with torch.autocast(device_type=params.DEVICE.type, enabled=torch.cuda.is_available()):
                    predictions = model(input_dict)
                    # compound loss BCE and Tversky
                    loss = self.compound_loss(
                        loss_1(predictions, label),
                        loss_2(predictions, label),
                        lambda_1=lambda_1,
                        lambda_2=lambda_2
                    )
                    print(f'DEBUGGUNG: loss: {loss.item():.3f}') 
                    # set NaNs=0
                    loss = loss.nan_to_num(0.0)
                    # don't count 0-loss batch in loss calculation  
                    if loss.item() !=0.0:
                        # update batch counter
                        num_batches += 1
                    epoch_train_loss += loss.item()

                # backpropagation
                scaler.scale(loss).backward()
                # convert gradient to real scale
                scaler.unscale_(optimizer)
                # gradient clipping -> clip gradients' norm to 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # every 100 iterations save model checkpoint
                if (iteration + 1) % 100 == 0:
                    self.update_checkpoint(
                        model, optimizer, epoch, iteration, epoch_train_loss
                    )

            # append average train loss for epoch
            if num_batches !=0:
                train_loss.append(epoch_train_loss / num_batches)
            else:
                train_loss.append(0.0)

            # VALIDATION
            model.eval()
            epoch_val_loss = 0.0
            num_batches = 0

            # iterate through the mini-batches
            for iteration, input_dict in enumerate(tqdm(val_dl, desc='Validation', leave=True)):
                with torch.no_grad():
                    # extract labels from input dictionary
                    label = input_dict['EO_label'].long()
                    # forward pass
                    predictions = model(input_dict)
                    # binary compound loss CE and Focal Tversky Loss
                    loss = self.compound_loss(
                        loss_1(predictions, label),
                        loss_2(predictions, label),
                        lambda_1=lambda_1,
                        lambda_2=lambda_2
                    )
                    print(f'DEBUGGUNG: loss: {loss.item()}') 
                    # set NaNs=0
                    loss = loss.nan_to_num(0.0)
                    # don't count 0-loss batch in loss calculation   
                    if loss.item() !=0.0:
                        # update batch counter
                        num_batches += 1
                    epoch_val_loss += loss.item()

            # append average val loss for epoch
            if num_batches !=0:
                val_loss.append(epoch_val_loss / num_batches)
            else:
                val_loss.append(0.0)

            # loss summary for training + validation epoch
            message = f'Training epoch {epoch + 1}: train loss: {train_loss[-1]:.4f}, validation loss: {val_loss[-1]:.4f}'
            pbar.set_description(message)
            self.logger.info(message)

            # early stopping
            # save validation loss if list val_los is empty
            if best_loss is None:
                best_loss = val_loss[-1]
                best_model_dict = deepcopy(model.state_dict())
            # if not empty: compare validation loss with best loss, save as best_loss if better
            else:
                if val_loss[-1] < best_loss:
                    best_loss = val_loss[-1]
                    best_model_dict = deepcopy(model.state_dict())
                    epochs_since_improvement = 0
                # if loss increases -> count early stopping patiences
                else:
                    epochs_since_improvement += 1
                    # if patience exceeded -> stop
                    if epochs_since_improvement >= patience:
                        # finish before max number of epochs
                        print('\rEarly stopping!', end='')
                        break

        # select and return best performing model for evaluation
        assert best_model_dict is not None
        model.load_state_dict(best_model_dict)
        model.eval()
        # returns trained model
        return model

    @torch.no_grad()
    def evaluate(
        self, 
        model, 
        test_dl, 
        vis=True,
        vis_method=utils.visualize_prediction_pastis
    ):
        '''
        Evaluation (test phase) of the fine-tuned segmentation model on the test set with 
        option to visualize model predictions as images.
        '''

        model.eval()
        # visualization lists to collect data (only visualized once per image)
        vis_preds, vis_labels, vis_eo_data = [], [], []
        # buffer for remainder
        buffer_labels, buffer_preds, buffer_eo_data = [], [], []
        # count pixels per image
        pixel_ctr = 0
        # metrics collection lists
        all_preds, all_labels = [], []
        # count images
        image_ctr = 0

        # iterate through the mini-batches
        for input_dict in tqdm(test_dl, desc='Testing', leave=True):
            # extract labels from input dictionary
            label = input_dict['EO_label'].long()
            # forward pass
            prediction = model(input_dict).float()

            # count pixels in buffer
            pixel_ctr += label.shape[0]
            # append predictions and labels for metrics calculation
            all_preds.append(prediction)
            all_labels.append(label)

            # visualize every full image prediction
            if vis:
                # append to visualisation list: predictions
                vis_preds.append(prediction)
                # labels
                vis_labels.append(label)
                # input Earth observation data
                vis_eo_data.append(input_dict['EO'])
                # if number of pixels of an image reached
                if pixel_ctr >= self.total_pixels:
                    # if something in buffer
                    if len(buffer_labels) > 0:
                        # append current collection to buffer
                        vis_labels = buffer_labels + vis_labels
                        vis_preds = buffer_preds + vis_preds
                        vis_eo_data = buffer_eo_data + vis_eo_data
                    # slice to image size
                    labels = torch.cat(vis_labels, dim=0)
                    preds = torch.cat(vis_preds, dim=0)
                    eo_data = torch.cat(vis_eo_data, dim=0)

                    # replace buffer by remainder
                    buffer_labels = [labels[self.total_pixels:]]
                    buffer_preds = [preds[self.total_pixels:]]
                    buffer_eo_data = [eo_data[self.total_pixels:]]
                    # reset collection lists
                    vis_preds, vis_labels, vis_eo_data = [], [], []

                    # visualize predictions
                    vis_method(
                        eo_data=eo_data[:self.total_pixels],
                        preds=preds[:self.total_pixels],
                        label=labels[:self.total_pixels],
                        title=f'test_{image_ctr}',
                        image_path=self.image_path,
                    )
                    # raise image count
                    image_ctr += 1
                    # count pixels in buffer
                    pixel_ctr = buffer_labels[0].shape[0]
        
        # concat to tensor
        all_labels = torch.cat(all_labels, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        # calculate metrics
        metrics = self.compute_metrics(all_preds, all_labels)

        return metrics
