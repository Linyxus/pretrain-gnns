device=$1
split=species

for seed in 0 1 2 3 4 5 6 7 8 9
do
  python finetune.py --model_file model_gin/grace_80.pth --split $split --filename gin_grace_80 --epochs 50 --device $device --runseed $seed --gnn_type gin --lr 1e-3
done