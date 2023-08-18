import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.helper.consoleLogs import log_curr_step_loss, log_final_eval, log_curr_epoch_loss, log_early_stopping, \
    log_model_training_info, CColors
from src.network.utiles import EarlyStopping


class FINTrainer:
    """
    This class combines model creation, training and evaluating
    """
    COLORS = CColors()

    def __init__(
            self,
            model,
            param: dict,
            train: np.array,
            test: np.array,
            train_crt: str = 'mae',
            eval_crt: str = 'mae',
            early_stop: EarlyStopping = None,
            verbose: bool = True):
        """
        parameters is a dict of format:
        dict(
                learning_rate=float,
                batch_size=int,
                epochs=int,
                activation=str,
                optimizer=str,
                weight_decay,
                momentum
        )
        """
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__verbose = verbose
        self.epochs = param['epochs']

        self.dl_train = DataLoader(dataset=train, batch_size=param['batch_size'], shuffle=True, drop_last=True)
        self.dl_test = DataLoader(dataset=test, batch_size=param['batch_size'], shuffle=True, drop_last=True)

        if early_stop:
            self.early_stopping = early_stop

        # Create an instance of the dynamic neural network
        self.model = model.to(self.__device)

        print(f'Running on device: {self.__device}')
        print('Model Architecture:')
        print(self.model)

        # Set criterion for training
        self.__criterion = nn.MSELoss()
        if train_crt == 'mae':
            self.__criterion = nn.L1Loss()

        # Set criterion for evaluation
        self.__criterion_eval = nn.MSELoss()
        if eval_crt == 'mae':
            self.__criterion_eval = nn.L1Loss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=param['learning_rate'],
            momentum=param['momentum'],
            weight_decay=param['weight_decay']
        )
        self.__scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=param['sched_ss'],
            gamma=param['sched_g']
        )
        if self.__verbose:
            log_model_training_info(
                self.__device,
                self.model,
                param['learning_rate'],
                param['momentum'],
                param['weight_decay']
            )

        self.loss_log = dict(
            test_loss=[],
            train_loss=[],
            train_loss_ep=[]
        )

    def fit_model(self):
        """
        Train the model on the training data and evaluate after each epoch on the test data.
        """

        for ep in range(self.epochs):
            self.model.eval()
            loss_val = self.eval_model()

            # Set model to Training mode
            self.model.train()
            loss_train = self.fit_one_epoch()

            self.__scheduler.step()

            self.loss_log['test_loss'].append(loss_val)
            self.loss_log['train_loss'].append(loss_train)
            if self.__verbose:
                lr = self.optimizer.param_groups[0]['lr']
                log_curr_epoch_loss((ep, self.epochs), loss_train, loss_val, lr)
            if self.early_stopping is not None:
                self.early_stopping(loss_train, loss_val)
                if self.early_stopping.early_stop:
                    log_early_stopping((ep, self.epochs))
                    break

        self.model.eval()
        loss_val = self.eval_model(nn.L1Loss())
        log_final_eval(loss_val)
        return loss_val

    def fit_one_epoch(self):
        """
        Fit for one epoch
        """
        running_loss = 0.0
        loss_ep = []

        for i, (inputs, targets) in enumerate(self.dl_train):
            if len(inputs.shape) > 2:
                inputs = torch.transpose(inputs, 0, 1)
            targets = targets.unsqueeze(1)

            # Forward pass
            y_pred = self.model(inputs)

            # Compute the loss
            loss = self.__criterion(y_pred, targets)
            loss_ep.append(loss.item())

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the running loss
            running_loss += loss.item()
            if len(self.dl_train) < 10:
                log_curr_step_loss((i, len(self.dl_train)), loss)
            else:
                if i % int(len(self.dl_train) / 10) == 0:
                    log_curr_step_loss((i, len(self.dl_train)), loss)
        self.loss_log['train_loss_ep'].append(loss_ep)
        return running_loss / len(self.dl_train)

    def eval_model(self, crt=None):
        """
        Evaluate the model on the test data
        """
        criterion = self.__criterion_eval
        if crt is not None:
            criterion = crt

        running_val_loss = 0.0
        for i, (inputs, targets) in enumerate(self.dl_test):
            if len(inputs.shape) > 2:
                inputs = torch.transpose(inputs, 0, 1)

            targets = targets.unsqueeze(1)

            # Forward pass
            y_pred = self.model(inputs)

            # Compute the loss
            loss = criterion(y_pred, targets)

            # Update the running loss
            running_val_loss += loss.item()

        # Compute average validation loss for the epoch
        return running_val_loss / len(self.dl_test)

    def save_model(self, fname):
        """
        Save the model
        """
        torch.save(self.model.state_dict(), fname)
