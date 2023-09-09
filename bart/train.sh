#Train code
export CUDA_VISIBLE_DEVICES=0
nohup python src/training/run_seq2seq.py \
        --bart_model facebook/bart-large \
        --data_dir ../../commongen/dataset \
        --train_src_file commongen.train.src_example.txt \
        --train_tgt_file commongen.train.tgt.txt \
        --dev_src_file commongen.dev.src_example.txt \
        --dev_tgt_file commongen.dev.tgt.txt \
        --output_dir /LOCAL2/name/result/matrix/bart/output/bart_large \
        --log_dir /LOCAL2/name/result/matrix/bart/log/bart_large \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --gradient_accumulation_steps 4 \
        --learning_rate 0.00002 \
        --num_train_epochs 5 \
        > train_bartlarge.log 2>&1 &