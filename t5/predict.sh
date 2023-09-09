export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_name_or_path /LOCAL2/name/result/matrix/t5/t5-large \
    --gradient_accumulation_steps=32 \
    --do_predict \
    --validation_file commongen.test.matrix.json \
    --test_file commongen.test.matrix.json \
    --text_column src \
    --summary_column tgt \
    --max_source_length=32 \
    --max_target_length=32 \
    --source_prefix "generate a sentence with these concepts: " \
    --output_dir mo_t5_large \
    --per_device_eval_batch_size=32 \
    --predict_with_generate