"""Early stopping 클래스 정의

TODO:
    * loss가 증가하는 방향으로 학습 시 옵션 설정 | 함수 따로 생성 | done
    * Docstring 작성
"""
import numpy as np
import logging

class EarlyStopper():
    """Early stoppiing 여부 판단
    
    Attributes:
        patience (int): loss가 줄어들지 않아도 학습할 epoch 수
        patience_counter (int): loss 가 줄어들지 않을 때 마다 1씩 증가
        best_loss (float): 최소 loss
        stop (bool): True 일 때 학습 중단

    """

    def __init__(self, patience: int, mode:str, logger:logging.RootLogger=None)-> None:
        """ 초기화
        
        Args:
            patience (int): loss가 줄어들지 않아도 학습할 epoch 수
            mode (str): max tracks higher value, min tracks lower value

        """
        self.patience = patience
        self.mode = mode
        self.logger = logger

        # Initiate
        self.patience_counter = 0
        self.stop = False
        self.best_loss = np.inf

        msg = f"Initiated ealry stopper, mode: {self.mode}, best score: {self.best_loss}, patience: {self.patience}"
        self.logger.info(msg) if self.logger else None
        
    def check_early_stopping(self, loss: float)-> None:
        """Early stopping 여부 판단

        Args:
            loss (float):

        Examples:
            
        Note:
            
        """  
        loss = -loss if self.mode == 'max' else loss  # get max value if mode set to max

        if loss > self.best_loss:
            # got better score
            self.patience_counter += 1

            msg = f"Early stopper, counter {self.patience_counter}/{self.patience}, best:{abs(self.best_loss)} -> now:{abs(loss)}"
            self.logger.info(msg) if self.logger else None
            
            if self.patience_counter == self.patience:
                msg = f"Early stopper, stop"
                self.logger.info(msg) if self.logger else None
                self.stop = True  # end

        elif loss <= self.best_loss:
            # got worse score
            self.patience_counter = 0
            self.best_loss = loss
            
            if self.logger is not None:
                self.logger.info(f"Early stopper, counter {self.patience_counter}/{self.patience}, best:{abs(self.best_loss)} -> now:{abs(loss)}")
                self.logger.info(f"Set counter as {self.patience_counter}")
                self.logger.info(f"Update best score as {abs(loss)}")

        else:
            print('debug')