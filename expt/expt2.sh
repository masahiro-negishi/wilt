cd $(dirname $0)
cd ../

MAX_JOBS=10

for depth in 3 4 5
do
    for lr in 0.0001 0.001 0.01
    do
        for temperature in 0.01 0.1 1 10 100
        do
            while (($(jobs -r | wc -l) >= $MAX_JOBS)); do
                sleep 1
            done
            python src/train.py --dataset_name "MUTAG" --depth $depth --loss_name "nce" \
            --batch_size "20" --n_epochs 1000 --lr $lr --save_interval 100 --seed 0 \
            --temperature $temperature &
        done
    done
done

wait