TRAIN_FILE=../../commongen/json_data/commongen.train.example.json
OUTPUT_DIR=/LOCAL2/name/result/ordered/bert/finetuned_models
CACHE_DIR=/LOCAL2/name/result/ordered/bert/cache

export CUDA_VISIBLE_DEVICES=1
nohup python run_seq2seq.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
  --model_type bert --model_name_or_path bert-base-uncased \
  --do_lower_case --max_source_seq_length 64 --max_target_seq_length 64 \
  --per_gpu_train_batch_size 32 --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 --num_warmup_steps 500 --num_training_epochs 30 --cache_dir ${CACHE_DIR} \
  > train.log 2>&1 &
