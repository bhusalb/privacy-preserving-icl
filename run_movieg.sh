#!/bin/bash

for i in 0 1 2 3 4 5 6 7
do
  seed=$i

  CUDA_VISIBLE_DEVICES=$i python src/privacy_preserving_icl/run_exp_movie_by_two.py \
    --num-private-train 380 \
    --num-valid 4 \
    --num-private-train-splits 380 \
    --num-test 1000 \
    --synth-seed $seed \
    --eval-seed $seed \
    --clipping-bound 10 \
    --temperature 1.0487702360295847 \
    --max-token-per-text 60 \
    --field-name "Genre" \
    --filename movieg.jsonl > output_movieg_${seed}.log
done