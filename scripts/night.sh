data_root="$1"
# source scripts/night.sh /data/shared/luojun

CUDA_VISIBLE_DEVICES=7 python train.py \
    --source-path "${data_root}/cityscapes/" \
    --synth-path "${data_root}/night_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/ACDC" \
    --target-type acdc_night \
    --experiment-name "${data_root}/runs/night1"

CUDA_VISIBLE_DEVICES=7 python train.py \
    --source-path "${data_root}/night_target" \
    --synth-path "${data_root}/night_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/ACDC" \
    --load "${data_root}/runs/night1/weights/checkpoint.pth.tar" \
    --target-type acdc_night \
    --stop_epoch 65 \
    --experiment-name "${data_root}/runs/night2"