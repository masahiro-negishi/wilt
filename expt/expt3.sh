cd $(dirname $0)
cd ../

MAX_JOBS=6

for depth in 3 4 5
do
    for lr in 0.0001 0.001 0.01
    do
        for margin in 0.01 0.1 1 10 100
        do
            while (($(jobs -r | wc -l) >= $MAX_JOBS)); do
                sleep 1
            done
            python src/train.py --dataset_name "MUTAG" --k_fold 5 --depth $depth --loss_name "triplet" \
            --batch_size "20" --n_epochs 1000 --lr $lr --save_interval 100 --seed 0 --clip_param_threshold "smallest_normal" --margin $margin &
        done
    done
done

wait