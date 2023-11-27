run at root folder
selfprocess.py (set to 1000 files)(processed files will be in arxiv_raw_stories)

run at src folder (cd src)
python preprocess.py -mode tokenize -raw_path ../arxiv_raw_stories -save_path ../testjson (tokenlize the data into testjson)
python preprocess.py -mode format_to_lines -raw_path ../testjson -save_path ../json_data -lower 
(process the tokenlized data into src and tgt json format so it can build bert pt data, shard_size can be set to other number (combine turn num of shard size into 1 file))
python preprocess.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data -oracle_mode greedy -n_cpus 4 (make bert data)


python train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 0  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 5000 -batch_size 3000 -decay_method noam -train_steps 20000-accum_count 2 -log_file ../logs/bert_classifier -use_interval true -warmup_steps 1000