save_dir=""
mkdir -p ${save_dir}
cd ./topology_vae/PyTorch_VAE
python run.py --save_dir ${save_dir} \
                          --use_pi True \
                          --categories chair \
                          --root_dir ../../ShapeNetCore.v2.PC15k/ \
                          --train_batch_size 64 \
                          --val_batch_size 64 \
                          --patch_size 64 \