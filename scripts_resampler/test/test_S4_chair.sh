save_dir=""
model_path=""
mkdir -p ${save_dir}

python test.py --distribution_type 'multi' \
    --dataroot ../../ShapeNetCore.v2.PC15k/ \
    --category chair \
    --use_resampler True\
    --model_type 'DiT-S/4' \
    --bs 8 \
    --voxel_size 32 \
    --experiment_name ${save_dir} \
    --model ${model_path} \
    --eval_path ${save_dir} \
    --num_classes 1 \
