import numpy as np
import torch
import torch.nn as nn

class PointGenerator:

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, featmap_size, stride=4):
        # returns [[x start, y start, stride], 
        #          [x1 start, y1 start, stride]
        #                       ...  
        #          [x31 start, y31 start, stride]
        #         ]
        # 입력된 feature map에 대한 center point와 stride를 지정해줌 
        feat_h, feat_w = featmap_size # (32, 32)
        shift_x = torch.arange(0., feat_w) * stride # [0, 1, 2, ..., 32]
        shift_y = torch.arange(0., feat_h) * stride # [0, 1, 2, ..., 32]
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        all_points = shifts

        return all_points

def get_points(featmap_sizes, img_metas, point_strides, point_generators):
    """Get points according to feature map sizes.
    Args:
        featmap_sizes (list[tuple]): Multi-level feature map sizes.
        img_metas (list[dict]): Image meta info.
    Returns:
        tuple: points of each image, valid flags of each image
    # feature map별로 point generator를 리스트 형태로 반환함 
    """
    # num_imgs = len(img_metas)
    num_imgs = img_metas
    num_levels = len(featmap_sizes)
    # num_levels = featmap_sizes

    # since feature map sizes of all images are the same, we only compute
    # points center for one time
    multi_level_points = []
    for i in range(num_levels):
        points = point_generators[i].grid_points(
            featmap_sizes[i], point_strides[i])
        multi_level_points.append(points)
    points_list = [[point.clone() for point in multi_level_points]
                    for _ in range(num_imgs)]

    return points_list 

def offset_to_pts(center_list, pred_list, point_strides, num_points):
    """Change from point offset to point coordinate.
    """
    pts_list = []
    for i_lvl in range(len(point_strides)):
        pts_lvl = []
        for i_img in range(len(center_list)):
            pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                1, num_points)
            pts_shift = pred_list[i_lvl][i_img]
            yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                -1, 2 * num_points)
            y_pts_shift = yx_pts_shift[..., 0::2]
            x_pts_shift = yx_pts_shift[..., 1::2]
            xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
            xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
            pts = xy_pts_shift * point_strides[i_lvl] + pts_center
            pts_lvl.append(pts)
        pts_lvl = torch.stack(pts_lvl, 0)
        pts_list.append(pts_lvl)

    return pts_list

def points2bbox(pts, 
                y_first=True, 
                transform_method="moment",
                moment_transfer=None, 
                moment_mul=None):
    """
    Converting the points set into bounding box.
    :param pts: the input points sets (fields), each points
        set (fields) is represented as 2n scalar.
    :param y_first: if y_fisrt=True, the point set is represented as
        [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
        represented as [x1, y1, x2, y2 ... xn, yn].
    :return: each points set is converting to a bbox [x1, y1, x2, y2].
    """
    pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                    ...]
    pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                    ...]
    if transform_method == 'minmax':
        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                            dim=1)
    elif transform_method == 'partial_minmax':
        pts_y = pts_y[:, :4, ...]
        pts_x = pts_x[:, :4, ...]
        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                            dim=1)
    elif transform_method == 'moment':
        pts_y_mean = pts_y.mean(dim=1, keepdim=True)
        pts_x_mean = pts_x.mean(dim=1, keepdim=True)
        pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
        pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
        moment_transfer = (moment_transfer * moment_mul) + (
            moment_transfer.detach() * (1 - moment_mul))
        moment_width_transfer = moment_transfer[0]
        moment_height_transfer = moment_transfer[1]
        half_width = pts_x_std * torch.exp(moment_width_transfer)
        half_height = pts_y_std * torch.exp(moment_height_transfer)
        bbox = torch.cat([
            pts_x_mean - half_width, pts_y_mean - half_height,
            pts_x_mean + half_width, pts_y_mean + half_height
        ],
                            dim=1)
    else:
        raise NotImplementedError
    return bbox

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

if __name__ == "__main__":

    ################### PointsGenerator 
    w, h = 32, 32 
    stride = 1
    """
    shift_x output : 
    tensor([  0.,   4.,   8.,  12.,  16.,  20.,  24.,  28.,  32.,  36.,  40.,  44.,
         48.,  52.,  56.,  60.,  64.,  68.,  72.,  76.,  80.,  84.,  88.,  92.,
         96., 100., 104., 108., 112., 116., 120., 124.])
    """
    shift_x = torch.arange(0., w) * stride 
    shift_y = torch.arange(0., h) * stride 
    
    shift_xx = shift_x.repeat(len(shift_y)) # 32x32=1024
    shift_yy = shift_y.view(-1, 1).repeat(1, len(shift_x)).view(-1)

    stride = shift_x.new_full((shift_xx.shape[0], ), stride)
    shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
    all_points = shifts

    ################### get_points 
    featmap_sizes = [(32, 32)]
    img_num = 1
    point_strides = [4]
    point_generators = [PointGenerator() for _ in point_strides]

    num_imgs = img_num 
    num_levels = len(featmap_sizes)

    multi_level_points = []
    for i in range(num_levels):
        points = point_generators[0].grid_points(featmap_sizes[0], point_strides[0])
        multi_level_points.append(points)
    points_list = [[point.clone() for point in multi_level_points]
                    for _ in range(num_imgs)]

    center_list = points_list 

    ################### offset_to_pts 
    pred_list = [torch.randn(1, 18, 32, 32)]
    pts_list = []
    for i_lvl in range(len(point_strides)):
        pts_lvl = []
        for i_img in range(len(center_list)):
            # pts_center (1024, 18)
            # 각 reppoints에 대한 center point 
            pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                1, 9
            )
            # single image feature map : (18, 32, 32)
            pts_shift = pred_list[i_lvl][i_img]
            # (1024, 18)
            yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                -1, 2 * 9
            )
            # y, x coordinates 
            y_pts_shift = yx_pts_shift[..., 0::2]
            x_pts_shift = yx_pts_shift[..., 1::2]
            # (1024, 9, 2)
            xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)

            # (1024, 18)
            xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
            pts = xy_pts_shift * point_strides[i_lvl] + pts_center 
            pts_lvl.append(pts)
        pts_lvl = torch.stack(pts_lvl, 0)
        pts_list.append(pts_lvl)

    pts_coordinate_preds_init = pts_list 

    ################### points2bbox 
    # (1024, 9, 2)
    y_first = False 
    transform_method = 'minmax'
    pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])

    # (1024, 9)
    pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
    pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
    if transform_method == 'minmax':
        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                            dim=1)
    print(bbox.shape)
    print(bbox)


