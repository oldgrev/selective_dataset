For working with datasets and picking out favourable training data. Specifically language models.

In the current iteration, this example uses 1M-GPT4-Augmented.parquet from Open-Orca/OpenOrca

Ultimately it goes from ~1 million rows to ~85,000 rows where:
- the response is always at least four times as long as the prompt
- the response is at least 10 characters long
- an analysis of the response is in the top 25% of the range of scores of the dataset. 
  - this was manually determined by looking at the distribution of scores in the dataset. which was 50% to 75%

The output is a json file for training in alpaca format. The system prompt is ignored. The goal is longer, more coherent responses, with assumption that "Less Is More for Alignment" holds true.

An outcome may be favourable responses.
