data_root="$1"
setting="$2"
# source img_gen/run.sh /data/shared/luojun

# if you have more than 1 gpu, you can specify their ids in the `gpu_ids` argument
python multigpu.py \
    --gpu_ids "0" \
    --type intermediate \
    --setting "$setting" \
    --image_root "${data_root}/cityscapes/leftImg8bit/train" \
    --label_root "${data_root}/cityscapes/gtFine/train" \
    --target_root "${data_root}/${setting}_intermediate"

python multigpu.py \
    --gpu_ids "0" \
    --type target \
    --setting "$setting" \
    --image_root "${data_root}/cityscapes/leftImg8bit/train" \
    --label_root "${data_root}/cityscapes/gtFine/train" \
    --target_root "${data_root}/${setting}_target"


target_types=("intermediate" "target")
sub_paths=("gtFine" "leftImg8bit/test" "leftImg8bit/val")

for type in "${target_types[@]}"; do
    for path1 in "${sub_paths[@]}"; do
        source_path="${data_root}/cityscapes/${path1}"
        target_path="${data_root}/${setting}_${type}/${path1}"

        if [ ! -e "${target_path}" ]; then
            mkdir -p "$(dirname "${target_path}")"
            echo "Creating: ${target_path}"
            ln -s "${source_path}" "${target_path}"
        else
            echo "Skipping: ${target_path}"
        fi
    done
done