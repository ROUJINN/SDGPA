import cv2
import numpy as np

labelpath = "/data3/roujin/cityscapes/gtFine/train/strasbourg/strasbourg_000000_019891_gtFine_color.png"

# synh_path ='/data3/roujin/snow_cityscapes_img2img/leftImg8bit/train/aachen/aachen_000013_000019_leftImg8bit.png'
synh_path = "/data3/roujin/zodi_cityscapes/leftImg8bit/train/trasbourg/strasbourg_000000_019891_leftImg8bit.png"
# synh_path = '7/out4.png'

gtpath = "/data3/roujin/cityscapes/leftImg8bit/train/trasbourg/strasbourg_000000_019891_leftImg8bit.png"

label = cv2.imread(labelpath)
synh_path = cv2.imread(synh_path)
print("synh Image Shape:", synh_path.shape)  # (H, W, C)
print("Label Image Shape:", label.shape)  # (H, W, C)
synh_path_resized = cv2.resize(synh_path, (label.shape[1], label.shape[0]))
alpha = 0.4  # Transparency, 0 is the complete original image, 1 is the complete labeled image
blended = cv2.addWeighted(synh_path_resized, 1 - alpha, label, alpha, 0)
cv2.imwrite("blended_result.jpg", blended)
