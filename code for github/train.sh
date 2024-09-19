workdir=$(cd $(dirname $0); pwd)
echo "workdir: "${workdir}
echo ""


WORLD_SIZE=4
deepspeed_config="config/deepspeed.json"
cluster_config="config/cluster.json"
echo "      WORLD_SIZE: "${WORLD_SIZE}
echo "deepspeed_config: "${deepspeed_config}
echo "  cluster_config: "${cluster_config}
echo ""


train_stage=1
output_path=""
model_path=""
data_path=""
output_dir=""
image_folder=""


args="--root_path ${output_dir} \
--data_path ${data_path} \
--model_path ${model_path} \
--image_folder ${image_folder} \
--seed 256 \
--deepspeed_config ${deepspeed_config} \
--train_stage ${train_stage}"
echo "args: "${args}
echo ""

run mdl --entry=train.py \
        --worker_count=${WORLD_SIZE}  \
        --user_params="$args" \
        --file.cluster_file=${cluster_config}
