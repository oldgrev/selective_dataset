analysis_out_file = "analysis_1.tsv"
dataset_file = '1M-GPT4-Augmented.parquet'

import json
import pandas as pd
df = pd.read_csv(analysis_out_file, sep="\t", header=0)
df = df.sort_values(by='score', ascending=False)

df_full = pd.read_parquet('1M-GPT4-Augmented.parquet', engine='pyarrow')
print("Dataset starting row count: " + str(len(df_full.index)))

df_full = df_full.set_index('id').join(df.set_index('id'))
df_full = df_full.reset_index()
df_full = df_full.dropna()

df_full['system_prompt'] = df_full['system_prompt'].str.strip()
df_full['question'] = df_full['question'].str.strip()
df_full['response'] = df_full['response'].str.strip()
print("Dataset scored row count: " + str(len(df_full.index)))

df_full = df_full[df_full['score'] > 0.5]
print("Results scored above 50%: " + str(len(df_full.index)))
df_full = df_full[df_full['score'] > 0.52]
print("Results scored above 52%: " + str(len(df_full.index)))
df_full = df_full[df_full['score'] > 0.54]
print("Results scored above 54%: " + str(len(df_full.index)))
df_full = df_full[df_full['score'] > 0.56]
print("Results scored above 56%: " + str(len(df_full.index)))
df_full = df_full[df_full['score'] > 0.58]
print("Results scored above 58%: " + str(len(df_full.index)))
df_full = df_full[df_full['score'] > 0.60]
print("Results scored above 60%: " + str(len(df_full.index)))
df_full = df_full[df_full['score'] > 0.62]
print("Results scored above 62%: " + str(len(df_full.index)))
df_full = df_full[df_full['score'] > 0.64]
print("Results scored above 64%: " + str(len(df_full.index)))
# df_full = df_full[df_full['score'] > 0.66]
# print("Results scored above 66%: " + str(len(df_full.index)))
# df_full = df_full[df_full['score'] > 0.68]
# print("Results scored above 68%: " + str(len(df_full.index)))
# df_full = df_full[df_full['score'] > 0.70]
# print("Results scored above 70%: " + str(len(df_full.index)))

json_list = []
for index, row in df_full.iterrows():
    json_list.append({
        "instruction": row['question'],
        "output": row['response']     
    })
with open("pruned_dataset.json", "w") as f:
    json.dump(json_list, f)
    f.close()




# Reference output:
# Dataset starting row count: 994896
# Dataset scored row count: 214493
# Results scored above 50%: 214493
# Results scored above 52%: 213880
# Results scored above 54%: 205838
# Results scored above 56%: 190254
# Results scored above 58%: 163752
# Results scored above 60%: 130617
# Results scored above 62%: 101565
# Results scored above 64%: 81203
# Results scored above 66%: 62556
# Results scored above 68%: 42911
# Results scored above 70%: 23226