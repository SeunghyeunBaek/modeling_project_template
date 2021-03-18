"""

TODO:
    * metric 함수 추가

"""
import numpy as np
import logging
import torch

class BatchTrainer():
    """Trainer
    
    Attribues:
        loss_function (Callable):
        optimizer (`optimizer`):
        model (`model`):
        device (str):
        logger (logging.RootLogger):
        metric_function (callable):
        train_loss_sum (float):
        train_loss_mean (float):
        validation_loss_sum (float):
        validation_loss_mean (float):
    """

    def __init__(self, model, optimizer, loss_function, device, metric_function, logger):
        
        # Train configuration
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.logger = logger
        self.metric_function = metric_function
        
        # Model
        self.best_loss = np.Inf

        # History - loss
        self.train_loss_sum = 0
        self.train_loss_mean = 0
        self.train_batch_loss_mean_list = list()
        self.train_batch_score_list = list()
        self.train_epoch_loss_mean_list = list()
        self.train_epoch_score_list = list()

        self.validation_loss_sum = 0
        self.validation_loss_mean = 0
        self.validation_batch_loss_mean_list = list()
        self.validation_batch_score_list = list()
        self.validation_epoch_loss_mean_list = list()
        self.validation_epoch_score_list = list()

        # History - predict
        self.train_target_list = list()
        self.train_target_pred_list = list()

        self.valdiation_target_list = list()
        self.validation_target_pred_list = list()

        # Output
        self.train_score = None
        self.validation_score = None


    def train_batch(self, dataloader, epoch_index, verbose=False, logging_interval=1):
        
        self.model.train()

        for batch_index, (image, target) in enumerate(dataloader):
            image, target = image.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Loss
            target_pred_proba = self.model(image)
            batch_loss_sum = self.loss_function(target_pred_proba, target, reduction='sum')
            batch_loss_mean = batch_loss_sum.item() / len(image)
            self.train_batch_loss_mean_list.append(batch_loss_mean)
            self.train_loss_sum += batch_loss_sum.item()

            # Metric
            target_pred_list = target_pred_proba.argmax(dim=1).cpu().tolist()
            targe_list = target.cpu().tolist()
            batch_score = self.metric_function(targe_list, target_pred_list)

            self.train_target_list += targe_list
            self.train_target_pred_list += target_pred_list
            self.train_batch_score_list.append(batch_score)

            # Update
            batch_loss_sum.backward()
            self.optimizer.step()

            # Log verbose
            if verbose & (batch_index % logging_interval == 0):
                msg = f"Train epoch {epoch_index} batch {batch_index}/{len(dataloader)}: {batch_index * len(image)}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                self.logger.info(msg) if self.logger else print(msg)

        self.train_loss_mean = self.train_loss_sum / len(dataloader.dataset)
        self.train_score = self.metric_function(self.train_target_list, self.train_target_pred_list)

        self.train_epoch_loss_mean_list.append(self.train_loss_mean)
        self.train_epoch_score_list.append(self.train_score)

        msg = f"Train epoch {epoch_index} completed, Mean loss: {self.train_loss_mean} Accuracy: {self.train_score}"
        self.logger.info(msg) if self.logger else print(msg)
        

    def validate_batch(self, dataloader, epoch_index, verbose=False, logging_interval=1):

        self.model.eval()

        with torch.no_grad():
            
            for batch_index, (image, target) in enumerate(dataloader):
                image, target = image.to(self.device), target.to(self.device)
                
                # Loss
                target_pred_proba = self.model(image)                
                batch_loss_sum = self.loss_function(target_pred_proba, target, reduction='sum')
                batch_loss_mean = batch_loss_sum.item() / len(image)
                self.validation_batch_loss_mean_list.append(batch_loss_mean)
                self.validation_loss_sum += batch_loss_sum

                # Metric
                target_pred_list = target_pred_proba.argmax(dim=1).cpu().tolist()
                target_list = target.cpu().tolist()
                batch_score = self.metric_function(target_list, target_pred_list)

                self.valdiation_target_list += target_list
                self.validation_target_pred_list += target_pred_list
                self.validation_batch_score_list.append(batch_score)

                # Log verbose
                if verbose & (batch_index % logging_interval == 0):
                    msg = f"Validation epoch {epoch_index} batch {batch_index}/{len(dataloader)}: {batch_index * len(image)}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} accuracy: {batch_score}"
                    self.logger.info(msg) if self.logger else print(msg)

            self.validation_loss_mean = self.validation_loss_sum / len(dataloader.dataset)
            self.validation_score = self.metric_function(self.valdiation_target_list, self.validation_target_pred_list)
            
            self.validation_epoch_loss_mean_list.append(self.validation_loss_mean)
            self.validation_epoch_score_list.append(self.validation_score)
            
            msg = f"Validation epoch {epoch_index} completed, Mean loss: {self.validation_loss_mean} Accuracy: {self.validation_score}"
            self.logger.info(msg) if self.logger else print(msg)
            
if __name__ == '__main__':
    pass