data_root="$1"
# source scripts/snow.sh /data/shared/luojun

CUDA_VISIBLE_DEVICES=1 python train.py \
    --source-path "${data_root}/cityscapes/" \
    --synth-path "${data_root}/snow_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/ACDC" \
    --target-type acdc_snow \
    --experiment-name "${data_root}/runs/snow1"

CUDA_VISIBLE_DEVICES=1 python train.py \
    --source-path "${data_root}/snow_target" \
    --synth-path "${data_root}/snow_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/ACDC" \
    --load "${data_root}/runs/snow1/weights/checkpoint.pth.tar" \
    --target-type acdc_snow \
    --stop_epoch 65 \
    --experiment-name "${data_root}/runs/snow2"