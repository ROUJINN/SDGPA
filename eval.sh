data_root="$1"
setting="$2"
# source eval.sh /data/shared/luojun snow

CUDA_VISIBLE_DEVICES=0 python eval.py \
    --cs_path "${data_root}/cityscapes/" \
    --acdc_path "${data_root}/ACDC" \
    --gta5_path "${data_root}/GTA5" \
    --weight "${data_root}/runs/${setting}2/weights/weights_65.pth.tar" \
    --setting "${setting}"