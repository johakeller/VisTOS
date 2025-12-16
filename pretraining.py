"""
Module to conduct entire pretraining. Uses model-checkpointing and logs training
performance to file. Entry point is init_pretraing().
"""

import logging
import math
import os
from datetime import datetime

import torch
import webdataset as wds
from torch import nn, optim
from tqdm import tqdm

import params
import utils
from dataset import pretraining_dataset

# set scaler for mixed precision training
scaler = torch.GradScaler(enabled=torch.cuda.is_available())


class PreTraining:
    """
    Central class to initiate and run pretraining and validation. Training is
    performed with streamed webdataset and with cosine-annealing scheduler. There
    is the additional option for model checkpointing. Mixed-precision training is
    used to save memory and computation cost. The pretrained model is saved in the
    output directory.
    """

    def __init__(
        self,
        epochs=params.EPOCHS,
        max_learning_rate=params.MAX_LR,
        min_learning_rate=params.MIN_LR,
        weight_decay=params.WEIGHT_DECAY,
        warmup_epochs=params.WARMUP_EPOCHS,
        vis_field_size=params.VIS_FIELDS[0],
    ):
        self.train_dataloader = None  # initialized in run_pretraining()
        self.validation_dataloader = None  # initialized in run_pretraining()
        self.test_dataloader = None  # initialized in run_pretraining()
        self.vis_field_size = vis_field_size
        # wrappend MSE loss (unmasked values ignored)
        self.mse = self.LossWrapper(nn.MSELoss())
        # wrapped Cross Entropy loss (unmasked values ignored)
        self.cross_entropy = self.LossWrapper(nn.CrossEntropyLoss())
        self.epochs = epochs
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.dynamic_world_loss_weight = params.DYNAMIC_WORLD_LOSS_WEIGHT
        self.logger = logging.getLogger("default")
        self.model_type = "att"

    class LossWrapper(nn.Module):
        """
        Wrapper class to select only masked inputs to feed to the loss function.
        Unmasked values are ignored. Returns loss calculated from masked values.
        """

        def __init__(self, loss: nn.Module):
            super().__init__()
            self.loss = loss

        def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
            assert len(pred) == len(true)
            if len(pred) == 0:
                # inputs are passed to the loss
                return torch.tensor(0).float().to(params.DEVICE)
            return self.loss(pred, true)

    def load_dataset(
        self,
        shuffle_on_load,
        masking_strategies,
        url,
        data_length,
    ):
        """
        Auxiliary function to initialize and return webdataset pipeline with
        passed parameters.
        """

        dataset = pretraining_dataset.PreTrainingDataset(
            masking_strategies=masking_strategies,
            shuffle=shuffle_on_load,
            length=data_length,
            vis_field_size=self.vis_field_size,
        )
        return dataset.create_pipeline(url=url)

    def init_pretraining(self, model, save=True, checkpointing=True, model_type="att"):
        """
        Starting point for module: initilizes the model, runs desired number of
        epochs of training, after the training epochs a validation epoch follows,
        save model optionally to file. Option to cache model periodically. Tries
        to load cached model from model cache first.
        """

        # init logger
        self.logger = utils.init_logger(
            name=f"pretraining_{model_type}_vf{self.vis_field_size}"
        )

        # info to file and to console output
        message = (
            f'\rPre-training VisTOS-{"attention" if model_type =="att" else "convolution"}, '
            f"hyperparameter: visual field size {self.vis_field_size}, "
            f"batch size {params.BATCH_SIZE}, "
            f'{datetime.now().strftime("%d-%m-%Y %H:%M")}\t'
        )
        print(message)
        self.logger.info(message)

        # model hyperparameters
        param_groups = self.param_groups_weight_decay(model, self.weight_decay)
        # set optimizer
        optimizer = optim.AdamW(
            param_groups, lr=self.max_learning_rate, betas=(0.9, 0.95), eps=1e-6
        )
        # default start epoch and iteration
        epoch = 0
        iteration = 0
        # set model type
        self.model_type = model_type

        # try loading model from cache
        if checkpointing:
            model, optimizer, epoch, iteration = self.load_checkpoint(
                model, optimizer, epoch, iteration
            )

        # define masking strategies
        masking_strategies = ["contiguous_timesteps", "random", "timesteps"]
        print(f"\rCreating dataset ...{params.EOL_SPACE}", end="")
        # create datasets
        train_dataset = self.load_dataset(
            shuffle_on_load=True,
            masking_strategies=masking_strategies,
            url=params.TRAIN_URL,
            data_length=params.TRAIN_DATA_LENGTH,
        )
        val_dataset = self.load_dataset(
            shuffle_on_load=False,
            masking_strategies=masking_strategies,
            url=params.VAL_URL,
            data_length=params.VAL_DATA_LENGTH,
        )
        print(f"\rCreating train dataloader ...{params.EOL_SPACE}", end="")
        # training dataloader
        self.train_dataloader = wds.WebLoader(
            train_dataset, batch_size=params.BATCH_SIZE
        )
        # validation dataloader
        print(f"\rCreating validation dataloader ...{params.EOL_SPACE}", end="")
        self.validation_dataloader = wds.WebLoader(
            val_dataset, batch_size=params.BATCH_SIZE
        )

        # start training
        model = self.training(
            model,
            optimizer,
            start_epoch=epoch,
            start_iteration=iteration,
            dataloader_len=params.TRAIN_DL_LENGTH,
        )
        # start validation
        self.validation(model, dataloader_len=params.VAL_DL_LENGTH)
        # optional saving of pretrained model to default output path
        if save:
            # save with size of visual field
            file_name = os.path.join(
                params.OUTPUT,
                f"{self.model_type}_model_weights_vf{self.vis_field_size}.pth",
            )
            torch.save(model.state_dict(), file_name)

    def load_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        iteration,
    ):
        """
        Loads a model from model checkpoint if one is available. Reproduces
        an returns cached model, optimizer, iteration, epoch calculated so far.
        """
        try:
            # try to fetch list of saved model checkpoints: check for model type
            # (attention/convolution) and correct visual field size
            old_checkpoints = [
                file
                for file in os.listdir(params.CACHE)
                if f"{self.model_type}_model_pt_{self.vis_field_size}" in file
            ]
            # if more than on suitable model found
            if len(old_checkpoints) > 1:
                print(
                    f"\rMore than one checkpoint available, loading last checkpoint: {old_checkpoints[-1]}.{params.EOL_SPACE}",
                    end="",
                )
                # fetch the last saved model
                old_checkpoint = old_checkpoints[-1]
            # only one model in list
            else:
                old_checkpoint = old_checkpoints[0]
            # load the selected model_checkpoint dictionary
            model_checkpoint = torch.load(
                os.path.join(params.CACHE, old_checkpoint), weights_only=True
            )
            # update model
            model.load_state_dict(model_checkpoint["model"])
            # update optimizer
            optimizer.load_state_dict(model_checkpoint["optimizer"])
            # update training epoch
            epoch = model_checkpoint["epoch"]
            # update training iteration
            iteration = model_checkpoint["iteration"]
        # cache is empty
        except IndexError:
            print(
                f"\rNo checkpoint found, starting from scratch.{params.EOL_SPACE}",
                end="",
            )
        # return cached model and training state
        return model.to(params.DEVICE), optimizer, epoch, iteration

    def update_checkpoint(self, model, optimizer, epoch, iteraton):
        """
        Saves current state of model and optimizer to checkpoint, removes old
        checkpoints.
        """
        # current state of training to dictionary: epoch, iteration, model, optimizer
        model_dict = {
            "epoch": epoch,
            "iteration": iteraton,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # remove old checkpoint with same visual field size and same model type
        old_checkpoints = [
            file
            for file in os.listdir(params.CACHE)
            if f"{self.model_type}_model_pt_{self.vis_field_size}" in file
        ]
        for file in old_checkpoints:
            os.remove(os.path.join(params.CACHE, file))

        # save new checkpoint
        model_path = os.path.join(
            params.CACHE,
            f'{self.model_type}_model_pt_{self.vis_field_size}_{datetime.now().strftime("%m_%d_%H")}.pth',
        )
        torch.save(model_dict, model_path)

    @staticmethod
    def get_dataloader_length(dataloader):
        """
        Returns the length of a Webdataset dataloader by counting the batches.
        Prints length to console.
        """
        length = 0
        for _ in dataloader:
            length += 1
            print(f"\rNumber batches: {length}{params.EOL_SPACE}", end="")
        return length

    @staticmethod
    def compound_loss(
        eo_prediction,
        dw_prediction,
        eo_loss,
        dw_loss,
        dynamic_world_loss_weight=params.DYNAMIC_WORLD_LOSS_WEIGHT,
    ):
        """
        Computes weighted compound loss as described by Tseng et al. (2023):

        Loss_total = Loss_MSE +Lambda* N_categorical/N_continuous*Loss_CE
        """

        # get number of masked values in EO data and in Dynamic World data
        num_eo_masked, num_dw_masked = len(eo_prediction), len(dw_prediction)
        with torch.no_grad():
            ratio = num_dw_masked / max(num_eo_masked, 1)
            # weight shouldn't be > 1
            weight = min(1, dynamic_world_loss_weight * ratio)

        return eo_loss + weight * dw_loss, num_eo_masked, num_dw_masked

    @staticmethod
    def param_groups_weight_decay(model, weight_decay=1e-5, no_weight_decay_list=()):
        """
        Groups model parameters into to lists of dictionaries, one with weight decay and one without
        weight decay. The no_weight_decay_list argument defines, which parameters are in no weight decay
        group, also bias and one dimensional params.

        Credit: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
        """

        # to set -> no repetitions
        no_weight_decay_list = set(no_weight_decay_list)
        decay = []
        no_decay = []
        # iterate through model parameters
        for name, param in model.named_parameters():
            # don't include untrainable parameters
            if not param.requires_grad:
                continue
            # if params 1D, bias or in no_weight_decay_list -> no weight decay
            if (
                param.ndim <= 1
                or name.endswith(".bias")
                or name in no_weight_decay_list
            ):
                no_decay.append(param)
            # weight decay for the remainder
            else:
                decay.append(param)
        # return list of 2 dictionaries: no weight decay params and weight decay params
        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    @staticmethod
    def adjust_learning_rate(
        optimizer,
        epoch,
        warmup_epochs,
        total_epochs,
        max_lr,
        min_lr,
    ):
        """
        Decay the learning rate with half-cycle cosine after warmup.
        """
        if epoch < warmup_epochs:
            lr = max_lr * epoch / warmup_epochs
        else:
            lr = min_lr + (max_lr - min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                )
            )
        for param_group in optimizer.param_groups:
            # set learnng rate in optimizer
            param_group["lr"] = lr
        return lr

    def training(
        self,
        model,
        optimizer,
        start_epoch=0,
        start_iteration=0,
        dataloader_len=0,
    ):
        """
        Training method for pretraining of the model. Uses a combined loss of cross-entropy and
        MSE and mixed-precision training. Saves model periodically (every 100 iterations) to model
        cache. Uses cosine-annealing schedule for learning rate.
        """

        model.train()
        # counts the number of batches by iteration if no length calculated
        if dataloader_len == 0:
            dataloader_len = self.get_dataloader_length(self.train_dataloader)
        print(
            f"\rTraining, dataloader length: {dataloader_len} batches of size {params.BATCH_SIZE}"
        )

        with tqdm(range(start_epoch, self.epochs), desc="Epoch") as tqdm_epoch:
            # iterate over epochs (starting at start epoch)
            for epoch in tqdm_epoch:
                # collect stats per epoch
                total_loss = 0
                total_eo_loss = 0
                total_dw_loss = 0
                total_val_num_eo_values_masked = 0
                total_val_num_dw_values_masked = 0
                num_batches = 0
                # iterate over mini-batches of epoch
                for iteration, input_dict in enumerate(
                    tqdm(self.train_dataloader, desc="Training", leave=True)
                ):
                    # skip until saved epoch and iteration:
                    if epoch == start_epoch and iteration < start_iteration:
                        print(
                            f"\rLoaded cached model: skip epoch {epoch+1} iteration {iteration}{params.EOL_SPACE}",
                            end="",
                        )
                        continue

                    # reset gradient
                    optimizer.zero_grad()
                    # learning rate via cosine annealing
                    _ = self.adjust_learning_rate(
                        optimizer,
                        iteration / dataloader_len + epoch,
                        self.warmup_epochs,
                        self.epochs,
                        self.max_learning_rate,
                        self.min_learning_rate,
                    )
                    # fetch data from input dictionary
                    eo_label = input_dict["EO_label"]
                    eo_mask = input_dict["EO_mask"].bool()
                    dw_label = input_dict["DW_label"].long()
                    dw_mask = input_dict["DW_mask"].bool()
                    # set SRTM False (expect for 1st step) so 11 unnecessary
                    # timesteps are ignored by the loss function (unmasked input)
                    eo_mask[:, 1:, params.CHANNEL_GROUPS["SRTM"]["idx"]] = False

                    # forward pass with mixed precision training (if cuda available)
                    with torch.autocast(
                        device_type=params.DEVICE.type,
                        enabled=torch.cuda.is_available(),
                    ):
                        eo_output, dw_output = model(input_dict)
                        # MSE for continuous EO values
                        eo_loss = self.mse(eo_output[eo_mask], eo_label[eo_mask])
                        # cross-entropy for Dynamic World
                        dw_loss = self.cross_entropy(
                            dw_output[dw_mask], dw_label[dw_mask]
                        )
                        # calculate compound loss
                        loss, num_eo_masked, num_dw_masked = self.compound_loss(
                            eo_output[eo_mask], dw_output[dw_mask], eo_loss, dw_loss
                        )
                    # backpropagation
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # stats: losses
                    total_loss += loss.item()
                    total_eo_loss += eo_loss.item() * num_eo_masked
                    total_dw_loss += dw_loss.item() * num_dw_masked
                    # count number of masked values
                    total_val_num_eo_values_masked += num_eo_masked
                    total_val_num_dw_values_masked += num_dw_masked
                    # update batch counter
                    num_batches += 1

                    # every 100 iterations save model checkpoint
                    if (iteration + 1) % 100 == 0:
                        # save model checkpoint
                        self.update_checkpoint(model, optimizer, epoch, iteration)

                # average losses
                avg_loss = total_loss / num_batches
                avg_eo_loss = total_eo_loss / max(
                    total_val_num_eo_values_masked, 1
                )  # avg. loss per masked EO element
                avg_dw_loss = total_dw_loss / max(
                    total_val_num_dw_values_masked, 1
                )  # avg. loss per masked DW element
                # loss summary for training
                message = f"Training epoch {epoch+1},\nTotal loss: {avg_loss:.4f}, \tDW loss: {avg_dw_loss:.4f}, \tEO loss {avg_eo_loss:.4f}"
                tqdm_epoch.write(message)
                self.logger.info(message)
        # return trained model
        return model

    def validation(self, model, dataloader_len=0):
        """
        Validation of a pretrained model. After each training epoch, outputs the validation
        loss to screen and logs it to file.
        """
        # evaluation mode
        model.eval()

        # counts the number of batches by iteration if no length calculated
        if dataloader_len == 0:
            dataloader_len = self.get_dataloader_length(self.validation_dataloader)
        print(
            f"\rValidation, dataloader length: {dataloader_len} batches of size {params.BATCH_SIZE}"
        )

        # collect stats
        total_loss = 0
        total_eo_loss = 0
        total_dw_loss = 0
        total_val_num_eo_values_masked = 0
        total_val_num_dw_values_masked = 0
        num_batches = 0
        # gradient off
        with torch.no_grad():
            # iterate over batches
            for input_dict in tqdm(
                self.validation_dataloader, desc="Validation", leave=True
            ):
                # fetch data from input dictionary
                eo_label = input_dict["EO_label"]
                eo_mask = input_dict["EO_mask"].bool()
                dw_label = input_dict["DW_label"].long()
                dw_mask = input_dict["DW_mask"].bool()
                # set SRTM False (expect for 1st step) so 11 unnecessary timesteps are ignored by the loss function (unmasked input)
                eo_mask[:, 1:, params.CHANNEL_GROUPS["SRTM"]["idx"]] = False
                # forward pass
                eo_output, dw_output = model(input_dict)
                # MSE for continuous EO values
                eo_loss = self.mse(eo_output[eo_mask], eo_label[eo_mask])
                # cross-entropy for Dynamic World
                dw_loss = self.cross_entropy(dw_output[dw_mask], dw_label[dw_mask])
                # calculate compound loss
                loss, num_eo_masked, num_dw_masked = self.compound_loss(
                    eo_output[eo_mask], dw_output[dw_mask], eo_loss, dw_loss
                )

                # stats
                total_loss += loss.item()
                total_eo_loss += eo_loss.item() * num_eo_masked
                total_dw_loss += dw_loss.item() * num_dw_masked
                # count number of masked values
                total_val_num_eo_values_masked += num_eo_masked
                total_val_num_dw_values_masked += num_dw_masked
                num_batches += 1

            # average losses
            avg_loss = total_loss / num_batches
            avg_eo_loss = total_eo_loss / max(total_val_num_eo_values_masked, 1)
            avg_dw_loss = total_dw_loss / max(total_val_num_dw_values_masked, 1)
            # loss summary for validation
            message = f"Validation,\nTotal loss: {avg_loss:.4f}, \tDW loss: {avg_dw_loss:.4f}, \tEO loss {avg_eo_loss:.4f}"
            tqdm.write(message)
            self.logger.info(message)
