# path of the fine-tuned checkpoint
MODEL_PATH=/LOCAL2/name/result/ordered/bert/finetuned_models/ckpt-31588
SPLIT=test
INPUT_JSON=.../../commongen/json_data/commongen.test.example.json

export CUDA_VISIBLE_DEVICES=1
python new_decode_seq2seq.py \
  --model_type bert --tokenizer_name bert-base-uncased --input_file ${INPUT_JSON} --split ${SPLIT} --do_lower_case \
  --model_path ${MODEL_PATH} --max_seq_length 128 --max_tgt_length 64 --batch_size 24 --beam_size 5 \
  --length_penalty 0 --forbid_duplicate_ngrams --mode s2s --forbid_ignore_word "."