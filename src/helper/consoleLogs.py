import logging
import os
import sys


class CColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


COLORS = CColors()

def log_loading_data(subj_id: int, progress: tuple[int, int]) -> None:
    logging.info(
        f"Loading EEG data"
        f": {progress[0] + 1} / {progress[1]} - Subject ID {subj_id}")


def log_n_samp_warning(total_subj: int, n_subj: int) -> None:
    logging.info(f"Warning"
                 f": Max number of subjects is {total_subj}. Defaulting from {n_subj} to {total_subj}.")


def log_error_target_not_available(target: str, avl_targets: list[str]):
    logging.info(f" ERROR:  Target {target} not in available targets {avl_targets}.")


def log_model_training_info(device, model, learning_rate, momentum, weight_decay):
    logging.info(f'Running on device: {device}')
    logging.info('Model Architecture:')
    logging.info(model)
    logging.info(f'Optimizer Parameters:')
    logging.info(f'    learning rate : {learning_rate}')
    logging.info(f'    momentum      : {momentum}')
    logging.info(f'    weight decay  : {weight_decay}')


def log_curr_epoch_loss(progress, loss_train, loss_val, lr):
    logging.info('')
    logging.info('====================================')
    logging.info(f'Epoch [{progress[0]}/{progress[1]}], '
                 f'Train: {loss_train:.3f}, '
                 f'Test: {loss_val:.3f}, '
                 f'lr: {lr}')
    logging.info('====================================')
    logging.info('')


def log_early_stopping(progress):
    logging.info('====================================')
    logging.info(f'Early stopping was triggerd: '
                 f'epoch {progress[0]} from {progress[1]}')
    logging.info('====================================')
    logging.info('')


def log_final_eval(loss_val):
    logging.info('====================================')
    logging.info(f'Final Evaluation MAE Loss: {loss_val:.3f}')
    logging.info('====================================')
    logging.info('')


def log_curr_step_loss(progress, loss):
    logging.info(f'Step  [{progress[0]}/{progress[1]}], '
                 f'Loss : {loss:.5f}')
