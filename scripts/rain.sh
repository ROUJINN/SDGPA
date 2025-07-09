data_root="$1"
# source scripts/rain.sh /data/shared/luojun

CUDA_VISIBLE_DEVICES=5 python train.py \
    --source-path "${data_root}/cityscapes/" \
    --synth-path "${data_root}/rain_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/ACDC" \
    --target-type acdc_rain \
    --experiment-name "${data_root}/runs/rain1"

CUDA_VISIBLE_DEVICES=5 python train.py \
    --source-path "${data_root}/rain_target" \
    --synth-path "${data_root}/rain_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/ACDC" \
    --load "${data_root}/runs/rain1/weights/checkpoint.pth.tar" \
    --target-type acdc_rain \
    --stop_epoch 65 \
    --experiment-name "${data_root}/runs/rain2"