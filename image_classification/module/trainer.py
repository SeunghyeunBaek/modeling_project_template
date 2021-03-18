"""
TODO:
    * metric 함수 추가

"""
from module.util import make_directory
import numpy as np
import logging
import torch
import csv
import os




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
        
        # History - loss
        self.train_batch_loss_mean_list = list()
        self.train_batch_score_list = list()

        self.validation_batch_loss_mean_list = list()
        self.validation_batch_score_list = list()

        # History - predict
        self.train_target_list = list()
        self.train_target_pred_list = list()

        self.valdiation_target_list = list()
        self.validation_target_pred_list = list()

        # Output
        self.train_score = 0
        self.train_loss_mean = 0
        self.train_loss_sum = 0
        
        self.validation_score = 0
        self.validation_loss_mean = 0
        self.validation_loss_sum = 0


    def train_batch(self, dataloader, epoch_index, verbose=False, logging_interval=1):
        
        self.model.train()

        for batch_index, (image, target) in enumerate(dataloader):
            image, target = image.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Loss
            target_pred_proba = self.model(image)
            batch_loss_mean = self.loss_function(target_pred_proba, target, reduction='mean')
            batch_loss_sum = batch_loss_mean.item() * len(image)
            self.train_batch_loss_mean_list.append(batch_loss_mean)
            self.train_loss_sum += batch_loss_sum

            # Metric
            target_pred_list = target_pred_proba.argmax(dim=1).cpu().tolist()
            targe_list = target.cpu().tolist()
            batch_score = self.metric_function(targe_list, target_pred_list)

            self.train_target_list += targe_list
            self.train_target_pred_list += target_pred_list
            self.train_batch_score_list.append(batch_score)

            # Update
            batch_loss_mean.backward()
            self.optimizer.step()

            # Log verbose
            if verbose & (batch_index % logging_interval == 0):
                msg = f"Train epoch {epoch_index} batch {batch_index}/{len(dataloader)}: {batch_index * len(image)}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                self.logger.info(msg) if self.logger else print(msg)

        self.train_loss_mean = self.train_loss_sum / len(dataloader.dataset)
        self.train_score = self.metric_function(self.train_target_list, self.train_target_pred_list)

        msg = f"Train epoch {epoch_index} Mean loss: {self.train_loss_mean} Accuracy: {self.train_score}"
        self.logger.info(msg) if self.logger else print(msg)
        

    def validate_batch(self, dataloader, epoch_index, verbose=False, logging_interval=1):

        self.model.eval()

        with torch.no_grad():
            
            for batch_index, (image, target) in enumerate(dataloader):
                image, target = image.to(self.device), target.to(self.device)
                
                # Loss
                target_pred_proba = self.model(image)                
                batch_loss_mean = self.loss_function(target_pred_proba, target, reduction='mean').item()
                batch_loss_sum = batch_loss_mean * len(image)
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
            
            msg = f"Validation epoch {epoch_index} Mean loss: {self.validation_loss_mean} Accuracy: {self.validation_score}"
            self.logger.info(msg) if self.logger else print(msg)


    def clear_history(self):
        """한 epoch 종료 후 history 초기화
            
            실험 당 BatchTrainer 를 한번만 생성하기 떄문에 한 epoch 이 끝나면 history 를 반드시 초기화 해야함
            list 초기화: BatchTrainer 에서 정의한 모든 list 형태의 attribute 는 배치마다 계속 element 를 적재하는 형태로 구현
            sum 초기화: BatchTrainer 에서 정의한 모든 _sum attribute 는 누적합계

            Examples:
                >>for epoch_index in range(EPOCH):
                >>    trainer.train_batch(dataloader=train_dataloader, epoch_index=epoch_index, verbose=False)
                >>    trainer.validate_batch(dataloader=validation_dataloader, epoch_index=epoch_index, verbose=False)
                >>    trainer.clear_history()  # 반드시 실행해야함
                
        """

        # History - loss
        self.train_batch_loss_mean_list = list()
        self.train_batch_score_list = list()

        self.validation_batch_loss_mean_list = list()
        self.validation_batch_score_list = list()

        # History - predict
        self.train_target_list = list()
        self.train_target_pred_list = list()

        self.valdiation_target_list = list()
        self.validation_target_pred_list = list()

        # Output
        self.train_score = 0
        self.train_loss_mean = 0
        self.train_loss_sum = 0
        
        self.validation_score = 0
        self.validation_loss_mean = 0
        self.validation_loss_sum = 0



class PerformanceRecorder():

    def __init__(self, serial: str, column_list: list, root_dir: str):
        """Recorder 초기화
            
            Args:
                serial (str):
                column_list (str):
                root_dir (str):
                record_dir (str):

            Note:
                Instance 생성 시 dir + serial/ 로 경로 생성

        """
        self.serial = serial
        self.column_list = column_list
        
        self.root_dir = root_dir
        self.record_dir = os.path.join(self.root_dir, self.serial)
        self.record_filepath = os.path.join(self.record_dir, 'record.csv')

        self.row_counter = 0

        make_directory(self.record_dir)


    def add_row(self, row):
        
        with open(self.record_filepath, newline='', mode='a') as f:
            writer = csv.writer(f)

            if self.row_counter == 0:
                writer.writerow(self.column_list)

            writer.writerow(row)
        
        self.row_counter += 1

if __name__ == '__main__':
    pass