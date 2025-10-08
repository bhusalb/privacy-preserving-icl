# export CUDA_VISIBLE_DEVICES=3


# for seed in 0 1 2 3 4 5 6 7 8 9
# do
# # # epsilon=1
# python src/privacy_preserving_icl/run_exp_agnews_by_two.py \
#     --num-private-train 450 \
#     --num-valid 4 \
#     --num-private-train-splits 450 \
#     --num-test 1000 \
#     --use-dp-prompts \
#     --synth-seed $seed \
#     --eval-seed $seed \
#     --clipping-bound 10 \
#     --temperature 1.0230255168894955 \
#     --max-token-per-text 60
#     --filename agnews.jsonl
# done

#!/bin/bash

for i in {0..7}
do
  seed=$i

  CUDA_VISIBLE_DEVICES=$i python src/privacy_preserving_icl/run_exp_dbpedia_by_two.py \
    --num-private-train 380 \
    --num-valid 4 \
    --num-private-train-splits 380 \
    --num-test 1000 \
    --synth-seed $seed \
    --eval-seed $seed \
    --clipping-bound 10 \
    --temperature 1.0487702360295847 \
    --max-token-per-text 60 \
    --filename dbpedia.jsonl > output_dbpedia_2_${seed}.log
done

# wait
