import pandas as pd

res_path = '/home/tobias/Schreibtisch/HPC-FIN/results/'

df = pd.read_csv(res_path + 'hpt_0.csv')

for i in range(1, 16):
    df_t = pd.read_csv(res_path + f'hpt_{i}.csv')
    df = pd.concat([df, df_t])

df = df.sort_values(by=['MAE score P2'])
print(df)
df.to_excel('./results_fin_3.xlsx')


