#Train's code
export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
    --model_name_or_path  t5-base \
    --gradient_accumulation_steps=3 \
    --do_train \
    --do_eval \
    --train_file commongen.train.matrix.json \
    --validation_file commongen.test.matrix.json \
    --text_column src \
    --summary_column tgt \
    --num_train_epochs 20 \
    --warmup_steps 400 \
    --learning_rate=5e-5 \
    --max_source_length=32 \
    --max_target_length=32 \
    --source_prefix "generate a sentence with these concepts: " \
    --output_dir /LOCAL2/name/result/matrix/t5/t5-base \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --predict_with_generate \
    > train_t5base.log 2>&1 &

