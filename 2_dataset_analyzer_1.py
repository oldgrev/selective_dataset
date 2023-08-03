# This script prunes the dataset on some arbitrary crud I made up.
# It also analyzes the dataset with a classifier.
# I don't touch the question column, because that can be as badly formatted as we like, just like us.

analysis_out_file = "analysis_1.tsv"
analysis_model_file = "salesken/query_wellformedness_score"
dataset_file = '1M-GPT4-Augmented.parquet'

# options
device = 0 # -1 for CPU
drop_responses_less_than_10 = True
drop_responses_shorter_than_question = True
drop_responses_less_than_4x_question = True
import pandas as pd
import os
import time
import csv
import re
from transformers import pipeline

df = pd.read_parquet(dataset_file, engine='pyarrow')
df = df[['id','question', 'response']]
df = df.dropna()

if not os.path.isfile(analysis_out_file):
    with open(analysis_out_file,"w") as f:
        f.write("id\tscore")
        f.close()

if drop_responses_less_than_10:
    df = df[df['response'].str.len() > 10]

if drop_responses_shorter_than_question:  # redundant, but a quick pruning pass
    df = df[df['response'].str.len() > df['question'].str.len()]

if drop_responses_less_than_4x_question:
    df = df[df['response'].str.len() / df['question'].str.len() > 4]

analyzed_ids = []
with open(analysis_out_file,"r") as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        analyzed_ids.append(line[0])
    f.close()

df = df[~df['id'].isin(analyzed_ids)]

classifier = pipeline("text-classification", model=analysis_model_file, top_k=None, device=0)

for index, row in df.iterrows():
    analysis = 0
    id = row['id']
    text = row['response']
    text_length = len(text)
    if text_length < 480:
        try:
            analysis = classifier(text)[0][0]['score']
            analysis = 0
            with open(analysis_out_file,"a") as f:
                f.write(str(id) + "\t" + str(analysis) + "\n")
                f.close()
        except Exception as e:
            print(f"{id}Error: {e}")
    else:
        text_chunks = text.split("\n")
        text_chunks_analysis = []
        for text_chunk in text_chunks:
            this_text_chunk = text_chunk + "\n"
            if len(this_text_chunk) > 479:
                this_text_chunk_chunks = re.split(r'(?<=[.!?]) +', this_text_chunk)
                text_chunks_analysis.extend(this_text_chunk_chunks)
            else:
                text_chunks_analysis.append(this_text_chunk)
        try:
            for text_chunk in text_chunks_analysis:
                analysis += classifier(text_chunk)[0][0]['score']
            analysis = analysis / len(text_chunks_analysis)
            with open(analysis_out_file,"a") as f:
                f.write(str(id) + "\t" + str(analysis) + "\n")
                f.close()
        except Exception as e:
            print(f"{id}Error: {e}")
            continue       

# there is now an analysis_out_file in the current folder that contains the id and score of each response
