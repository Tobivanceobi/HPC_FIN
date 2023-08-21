import numpy as np

from src.helper.pickleLoader import save_object


num_of_proc = [i for i in range(0, 16)]

learning_rate = np.linspace(0.0001, 0.01)
batch_size = [32, 64, 128, 256, 512]

hidden_sizes = [
    [300],
    [250],
    [100],
    [300, 100],
    [250, 50],
    [200, 50],
]
epochs = 30
activation = ['relu', 'tanh', 'sigmoid']
optimizer = ['sgd']
dropout_p = [0.2, 0.3, 0.4]
weight_decay = [0.0001, 0.001, 0.01]
momentum = [0.1, 0.5, 0.9]


def create_hpt_space():
    hp_space = []
    for lr_idx in range(len(learning_rate)):
        for bs in batch_size:
            for hs in hidden_sizes:
                for act in activation:
                    for opt in optimizer:
                        for wd in weight_decay:
                            for m in momentum:
                                for d in dropout_p:
                                    conf = dict(
                                        hidden_sizes=hs,
                                        learning_rate=learning_rate[lr_idx],
                                        batch_size=bs,
                                        epochs=epochs,
                                        activation=act,
                                        optimizer=opt,
                                        output_size=1,
                                        dropout_p=d,
                                        weight_decay=wd,
                                        momentum=m,
                                        sched_ss=20,
                                        sched_g=0.9
                                    )
                                    hp_space.append(conf)

    print('Number of parameter settings: ', len(hp_space))
    task_arr = np.array_split(hp_space, len(num_of_proc))
    for i in num_of_proc:
        print(f'Number of parameter settings for {i}: {len(task_arr[i])}')
        save_object(task_arr[i], f'./hp_params/hptSpace_{i}')


create_hpt_space()
