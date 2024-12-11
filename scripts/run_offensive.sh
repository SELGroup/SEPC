SEED="0 1 2 3 4"
for seed in ${SEED[@]}
do
python3 -u main.py \
    --fine_tune_task tweeteval_offensive \
    --task_type cls \
    --pretrained_model_path ptms/roberta-base \
    --output_dir outputs/spc/tweeteval_offensive_roberta/tweeteval_offensive \
    --seed 0 \
    --var_weight 0.1 \
    --clu_weight 0.1 \
    --dropout 0 \
    --max_length 128 \
    --weight_decay 0 \
    --warmup_ratio 0.1 \
    --bs 128 \
    --epoch 25 \
    --patience 5 \
    --dataset_percentage 1 \
    --lr 5e-5 \
    --batch_sampling_flag \
    --normalize_flag
done