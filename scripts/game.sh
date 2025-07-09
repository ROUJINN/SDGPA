data_root="$1"
# source scripts/game.sh /data/shared/luojun

CUDA_VISIBLE_DEVICES=4 python train.py \
    --source-path "${data_root}/cityscapes/" \
    --synth-path "${data_root}/game_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/GTA5" \
    --target-type gta5 \
    --experiment-name "${data_root}/runs/game1"

CUDA_VISIBLE_DEVICES=4 python train.py \
    --source-path "${data_root}/game_target" \
    --synth-path "${data_root}/game_intermediate/leftImg8bit/train/" \
    --target-path "${data_root}/GTA5" \
    --load "${data_root}/runs/game1/weights/checkpoint.pth.tar" \
    --target-type gta5 \
    --stop_epoch 65 \
    --experiment-name "${data_root}/runs/game2"