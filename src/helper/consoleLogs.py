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
    sys.stdout.write('\r')
    logging.info(
        f"{COLORS.OKGREEN}Loading {COLORS.BOLD}EEG data{COLORS.ENDC}"
        f": {progress[0] + 1} / {progress[1]} - Subject ID {subj_id}", end='')


def log_n_samp_warning(total_subj: int, n_subj: int) -> None:
    print('')
    print(f"{COLORS.WARNING}Warning{COLORS.ENDC}"
          f": Max number of subjects is {total_subj}. Defaulting from {n_subj} to {total_subj}.")
    print('')


def log_error_target_not_available(target: str, avl_targets: list[str]):
    print(f"{COLORS.FAIL} ERROR: {COLORS.ENDC} Target {target} not in available targets {avl_targets}.")


def log_model_training_info(device, model, learning_rate, momentum, weight_decay):
    print(f'Running on device: {device}')
    print('Model Architecture:')
    print(model)
    print(f'{COLORS.OKBLUE}Optimizer Parameters:{COLORS.ENDC}')
    print(f'    {COLORS.OKCYAN}learning rate {COLORS.ENDC}: {learning_rate}')
    print(f'    {COLORS.OKCYAN}momentum      {COLORS.ENDC}: {momentum}')
    print(f'    {COLORS.OKCYAN}weight decay  {COLORS.ENDC}: {weight_decay}')


def log_curr_epoch_loss(progress, loss_train, loss_val, lr):
    print('')
    print('====================================')
    print(f'{COLORS.BOLD}Epoch{COLORS.ENDC} [{progress[0]}/{progress[1]}], '
          f'{COLORS.OKBLUE}Train{COLORS.ENDC}: {loss_train:.3f}, '
          f'{COLORS.OKGREEN}Test{COLORS.ENDC}: {loss_val:.3f}, '
          f'{COLORS.OKCYAN}lr{COLORS.ENDC}: {lr}')
    print('====================================')
    print('')


def log_early_stopping(progress):
    print('====================================')
    print(f'{COLORS.WARNING}Early stopping was triggerd{COLORS.ENDC}: '
          f'epoch {progress[0]} from {progress[1]}')
    print('====================================')
    print('')


def log_final_eval(loss_val):
    print('====================================')
    print(f'{COLORS.OKGREEN}Final Evaluation MAE Loss{COLORS.ENDC}: {loss_val:.3f}')
    print('====================================')
    print('')


def log_curr_step_loss(progress, loss):
    print(f'{COLORS.BOLD}Step{COLORS.ENDC}  [{progress[0]}/{progress[1]}], '
          f'{COLORS.OKBLUE}Loss{COLORS.ENDC} : {loss:.5f}')
