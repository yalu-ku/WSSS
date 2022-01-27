import os 
import sys
sys.path.append(os.getcwd())
from models.reppoints_utils import (PointGenerator, get_points, offset_to_pts,
                                    points2bbox)
from utils.transforms import transforms 
from PIL import Image 
from models.vgg_reppoints import vgg16
import cv2 

mean_vals = [0.485, 0.456, 0.406]
std_vals = [0.229, 0.224, 0.225]

img_name = "/home/junehyoung/code/wsss_baseline/cls/figure/dog.png"

tsfm_train = transforms.Compose([transforms.Resize(384),  
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                    transforms.RandomCrop(320),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])
model = vgg16(pretrained=True)
img = Image.open(img_name).convert('RGB')
input = tsfm_train(img).unsqueeze(0)
# print(input.shape)
logit, reppoints = model(input)

featmap_sizes = [reppoints.shape[-2:]] # (32, 32)
img_num = reppoints.shape[0] # 1 
point_strides = [4]
point_generators = [PointGenerator() for _ in point_strides]
num_points = 9
transform_method = "minmax"

center_list = get_points(featmap_sizes, img_num, point_strides, point_generators)
pts_coordinate_preds_init = offset_to_pts(center_list, [reppoints], point_strides, num_points)
print(pts_coordinate_preds_init[0].shape)
bbox_pred_init = points2bbox(
        pts_coordinate_preds_init[0].reshape(-1, 2 * 9), 
        y_first=False, transform_method=transform_method)
reppoints = pts_coordinate_preds_init[0].squeeze(0)
img = cv2.imread(img_name)

for pts in reppoints:
    x_coords = pts[0::2]
    x_coords = [0 if i < 0 else int(i) for i in x_coords]
    y_coords = pts[1::2]
    y_coords = [0 if i < 0 else int(i) for i in y_coords]
    if 0 in x_coords or 0 in y_coords:
        continue 
    for x, y in zip(x_coords, y_coords):
        cv2.circle(img, (x, y), radius=0, color=(0, 0, 255), thickness=3)
    
cv2.imwrite("rep_pointed_dog.png", img)

img = cv2.imread(img_name)

for bbox in bbox_pred_init:
    x1, y1, x2, y2 = bbox 
    if x1 < 0:
        continue 
    if y1 < 0:
        continue 
    if x2 < 0:
        continue 
    if y2 < 0:
        continue 
    x1 = max(0, int(x1.item()))
    y1 = max(0, int(y1.item()))
    x2 = max(0, int(x2.item()))
    y2 = max(0, int(y2.item()))
    print(x1, y1, x2, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.imwrite("pseudo_boxed_dog.png", img)

