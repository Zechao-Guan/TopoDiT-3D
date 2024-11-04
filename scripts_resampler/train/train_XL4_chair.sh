save_dir=""
mkdir -p ${save_dir}

python train.py --distribution_type 'multi' \
    --dataroot ../../ShapeNetCore.v2.PC15k/ \
    --category chair \
    --experiment_name ${save_dir} \
    --use_resampler True \
    --model_type 'DiT-XL/4' \
    --bs 512 \
    --voxel_size 32 \
    --lr 1e-4 \
    --num_classes 1 \
    --use_tb \
