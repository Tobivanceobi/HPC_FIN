import numpy as np

from src.helper.pickleLoader import save_object


num_of_proc = [i for i in range(0, 16)]

learning_rate = np.linspace(0.000001, 0.0001, 20)
batch_size = [64, 128, 256, 512]

hidden_sizes = [
    [400],
    [250],
    [400, 250],
    [250, 150],
    [400, 200, 100],
    [250, 150, 75],
    [400, 200, 100, 100],
    [250, 150, 75, 50],
    [400, 200, 100, 50, 20],
    [250, 150, 75, 50, 20],
    [400, 200, 100, 50, 20, 10],
    [250, 150, 75, 50, 20, 10],
    [400, 200, 100, 50, 20, 50, 10],
    [250, 150, 75, 50, 20, 10, 5]
]
epochs = 10
activation = ['relu', 'tanh', 'sigmoid']
optimizer = ['sgd', 'adam']
dropout_p = [0.1, 0.2, 0.3, 0.4]
weight_decay = [0.0001, 0.001, 0.01]
momentum = [0.1, 0.3, 0.6, 0.9]


def create_hpt_space():
    hp_space = []
    for lr_idx in range(len(learning_rate)):
        for bs in batch_size:
            for hs in hidden_sizes:
                for act in activation:
                    for opt in optimizer:
                        conf = dict(
                            hidden_sizes=hs,
                            learning_rate=learning_rate[lr_idx],
                            batch_size=bs,
                            epochs=epochs,
                            activation=act,
                            optimizer=opt,
                            output_size=1,
                            dropout_p=dropout_p,
                            weight_decay=weight_decay,
                            momentum=momentum,
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
