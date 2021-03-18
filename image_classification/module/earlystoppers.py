"""Early stopping 클래스 정의

TODO:
    * loss가 증가하는 방향으로 학습 시 옵션 설정
"""
import numpy as np
import logging
import torch

class LossEarlyStopper():
    """
    
    Attributes:
        patience (int): loss가 줄어들지 않아도 학습할 epoch 수
        verbose (bool): 로그 출력 여부, True 일 때 로그 출력
        weight_path (str): checkpoint 저장 경로

        patience_counter (int): loss 가 줄어들지 않을 때 마다 1씩 증가
        min_loss (float): 최소 loss
        stop (bool): True 일 때 학습 중단

    """

    def __init__(self, patience: int, weight_path: str, verbose: bool, logger:logging.RootLogger=None)-> None:
        """ 초기화
        
        Args:
            patience (int): loss가 줄어들지 않아도 학습할 epoch 수
            weight_path (str): weight 저장경로
            verbose (bool): 로그 출력 여부, True 일 때 로그 출력

        """
        self.patience = patience
        self.weight_path = weight_path
        self.verbose = verbose

        self.patience_counter = 0
        self.min_loss = np.Inf
        self.logger = logger
        self.stop = False


    def check_early_stopping(self, loss: float, model: "model")-> None:
        """Early stopping 여부 판단

        Args:
            loss (float):
            model (`model`):

        Examples:
            
        Note:
            
        """  

        if self.min_loss == np.Inf:
            self.min_loss = loss
            self.save_checkpoint(loss=loss, model=model)

        elif loss > self.min_loss:
            self.patience_counter += 1
            msg = f"Early stopping counter {self.patience_counter}/{self.patience}"

            if self.patience_counter == self.patience:
                self.stop = True

            if self.verbose:
                self.logger.info(msg) if self.logger else print(msg)
                
        elif loss <= self.min_loss:
            msg = f"Validation loss decreased {self.min_loss} -> {loss}"
            self.min_loss = loss
            self.save_checkpoint(loss=loss, model=model)

            if self.verbose:
                self.logger.info(msg) if self.logger else print(msg)


    def save_checkpoint(self, loss: float, model: "model")-> None:
        """Weight 저장

        Args:
            loss (float): validation loss
            model (`model`): model
        
        """
        msg = f"Weight saved: {self.weight_path}"
        torch.save(model.state_dict(), self.weight_path)
        self.min_loss = loss

        if self.verbose:
            self.logger.info(msg) if self.logger else print(msg)
