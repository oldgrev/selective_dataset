For working with datasets and picking out favourable training data. Specifically language models.

In the current iteration, this example uses 1M-GPT4-Augmented.parquet from Open-Orca/OpenOrca

Ultimately it goes from ~1 million rows to ~44,000 rows where:
- the response is always at least four times as long as the prompt
- the response is at least 10 characters long
- an analysis of the response is in the top 25% of the range of scores of the dataset. 
  - this was manually determined by looking at the distribution of scores in the dataset. which was 50% to 75%

The output is a json file for training in alpaca format. The system prompt is ignored. The goal is longer, more coherent responses, with assumption that "Less Is More for Alignment" holds true.

An outcome may be favourable responses.


Training.

	2023-08-05 18:18:35 INFO:Training complete, saving...
	
	2023-08-05 18:18:35 INFO:Training complete!
	
	2023-08-05 18:20:07 INFO:Loading JSON datasets...
	
	(Model has been modified by previous training, it needs to be reloaded...)
	
	2023-08-05 18:20:09 INFO:Loading TheBloke_Llama-2-13B-fp16...
	
	2023-08-05 18:20:09 WARNING:Using the following 4-bit params: {'load_in_4bit': True, 'bnb_4bit_compute_dtype': torch.float16, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True}
	
	Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████| 3/3 [00:07<00:00,  2.59s/it]
	
	2023-08-05 18:20:17 WARNING:models/TheBloke_Llama-2-13B-fp16/special_tokens_map.json is different from the original LlamaTokenizer file. It is either customized or outdated.
	
	2023-08-05 18:20:17 INFO:Loaded the model in 8.07 seconds.
	
	
	
	Model reloaded OK, continue with training.
	
	2023-08-05 18:20:17 INFO:Getting model ready...
	
	2023-08-05 18:20:17 INFO:Prepping for training...
	
	2023-08-05 18:20:17 INFO:Creating LoRA model...
	
	2023-08-05 18:20:22 INFO:Starting training...
	
	Training 'llama' model using (q, v) projections
	
	Trainable params: 26,214,400 (0.3914 %), All params: 6,698,193,920 (Model: 6,671,979,520)
	
	2023-08-05 18:20:22 INFO:Log file 'train_dataset_sample.json' created in the 'logs' directory.
	
	Step: 159 {'loss': 1.8333, 'learning_rate': 2.9911242603550295e-05, 'epoch': 0.01}
	
	Step: 319 {'loss': 1.7755, 'learning_rate': 2.9556213017751482e-05, 'epoch': 0.03}
	
	Step: 479 {'loss': 1.6629, 'learning_rate': 2.911242603550296e-05, 'epoch': 0.04}
	
	Step: 639 {'loss': 1.6121, 'learning_rate': 2.8668639053254437e-05, 'epoch': 0.06}
