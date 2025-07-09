data_root="$1"
# source scripts/fog.sh /data/shared/luojun

CUDA_VISIBLE_DEVICES=6 python train.py \
    --source-path "${data_root}/cityscapes/" \
    --synth-path "${data_root}/fog_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/ACDC" \
    --target-type acdc_fog \
    --experiment-name "${data_root}/runs/fog1"

CUDA_VISIBLE_DEVICES=6 python train.py \
    --source-path "${data_root}/fog_target" \
    --synth-path "${data_root}/fog_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/ACDC" \
    --load "${data_root}/runs/fog1/weights/checkpoint.pth.tar" \
    --target-type acdc_fog \
    --stop_epoch 65 \
    --experiment-name "${data_root}/runs/fog2"