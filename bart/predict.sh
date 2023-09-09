#Generate code
export CUDA_VISIBLE_DEVICES=0
python src/training/decode_seq2seq.py \
            --bart_model facebook/bart-large \
            --input_file ../../commongen/dataset/commongen.test.src_ordered.txt \
            --model_recover_path /LOCAL2/name/result/ordered/bart/output/bart_large/best_model/model.best.bin \
            --output_dir /home/name/bart_large/test/ \
            --output_file example_ordering_bart_large.txt \
            --batch_size 100