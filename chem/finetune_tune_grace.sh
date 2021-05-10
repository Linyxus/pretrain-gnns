#### GIN fine-tuning
device=$2
split=scaffold

### for GIN
for runseed in 0 1 2 3 4 5 6 7 8 9
do
for dataset in bace bbbp clintox hiv muv sider tox21 toxcast
do
python finetune.py --input_model_file model_gin/grace.pth --split $split --filename ${dataset}/gin_grace --device $device --runseed $runseed --gnn_type gin --dataset $dataset
done
done