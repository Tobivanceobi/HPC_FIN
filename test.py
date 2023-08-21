import pandas as pd

res_path = '/home/tobias/Schreibtisch/FIN-Cache/FIN-Results/run_2/'

df = pd.read_csv(res_path + 'hpt_0.csv')

for i in range(1, 16):
    df_t = pd.read_csv(res_path + f'hpt_{i}.csv')
    df = pd.concat([df, df_t])
print(df)
df.to_excel('./results_fin_2.xlsx')


