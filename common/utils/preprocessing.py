import cv2
import math
import random
import numpy as np
from config import cfg


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)
    if order=='RGB':
        img = img[:,:,::-1].copy()
    img = img.astype(np.float32)
    return img


def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width * 1.2
    xmax = x_center + 0.5*width * 1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height * 1.2
    ymax = y_center + 0.5*height * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, img_width, img_height, data_split, dataset='MSCOCO'):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    if data_split == 'train':
        scale = 0.25
    else:
        scale = 0.0
    
    bbox[2] = w * (1 + scale)
    bbox[3] = h * (1 + scale)
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox


def match_dp_bbox(bbox, dp_boxes, img_shape):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_shape[1] - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_shape[0] - 1, y1 + np.max((0, h - 1))))
    area1 = (x2 - x1) * (y2 - y1)
        
    iou_max = 0
    idx_matched = 0
    bbox_matched = None
    for i, bb in enumerate(dp_boxes):
        area2 = (bb[2] - bb[0]) * (bb[3] - bb[1])
        x_left = max(x1, bb[0])
        y_top = max(y1, bb[1])
        x_right = min(x2, bb[2])
        y_bottom = min(y2, bb[3])
        if area1 <= 0 or area2 <= 0 or x_right < x_left or y_bottom < y_top:
            continue
        intersection = (x_right - x_left) * (y_bottom - y_top)
        iou = intersection / float(area1 + area2 - intersection)
        if iou > iou_max:
            iou_max = iou
            idx_matched = i
            bbox_matched = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
    return idx_matched, iou_max, bbox_matched


def get_aug_config(exclude_flip):
    scale_factor = 0 if cfg.no_augment else 0.25
    rot_factor = 0 if cfg.no_augment else 30
    color_factor = 0 if cfg.no_augment else 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -1.0, 1.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip or cfg.no_augment:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    return scale, rot, color_scale, do_flip


def augmentation(img, dp_img, bbox, data_split, exclude_flip=False):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config(exclude_flip)
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False
    if cfg.occ_augment:
        cent_x = bbox[0] + bbox[2]/2 + np.random.uniform(-0.1, 0.1) * bbox[2]
        cent_y = bbox[1] + bbox[3]/2 + np.random.uniform(-0.5, 0.5) * bbox[3]
        w = np.random.uniform(0, 0.3) * bbox[2]
        h = np.random.uniform(0, 0.3) * bbox[2]
        x1, x2 = max(int(cent_x-w/2), 0), min(int(cent_x+w/2), img.shape[1])
        y1, y2 = max(int(cent_y-h/2), 0), min(int(cent_y+h/2), img.shape[0])
        # img[y1:y2, x1:x2, :] = np.array([128, 128, 128])[None,None,:]
        if x2 > x1 and y2 > y1:
            img[y1:y2, x1:x2, :] = img[:y2-y1, :x2-x1, :]
        if do_flip:
            occ_bbox = [img.shape[1]-x2, y1, img.shape[1]-x1, y2]
        else:
            occ_bbox = [x1, y1, x2, y2]
    else:
        occ_bbox = [0,0,0,0]
    img, dp_img, trans, inv_trans = generate_patch_image(img, dp_img, bbox, scale, rot, do_flip, cfg.input_img_shape)
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    return img, dp_img, trans, inv_trans, rot, do_flip, occ_bbox


def generate_patch_image(cvimg, cvdp_img, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    dp_img = cvdp_img.copy()
    img_height, img_width, img_channels = img.shape
    
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        dp_img = dp_img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR).astype(np.float32)
    dp_img_patch = cv2.warpAffine(dp_img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_NEAREST).astype(np.uint8)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, dp_img_patch, trans, inv_trans


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def check_occ(coord, occ_bbox):
    occ_x = ((coord[:,0] >= occ_bbox[0]) * (coord[:,0] < occ_bbox[2])).reshape(-1,1).astype(np.float32)
    occ_y = ((coord[:,1] >= occ_bbox[1]) * (coord[:,1] < occ_bbox[3])).reshape(-1,1).astype(np.float32)
    return 1 - occ_x * occ_y


def check_trunc(coord, output_shape, is_3D=True, combine=False):
    if is_3D:
        trunc_x = ((coord[:,0] >= 0) * (coord[:,0] < output_shape[2])).reshape(-1,1).astype(np.float32)
        trunc_y = ((coord[:,1] >= 0) * (coord[:,1] < output_shape[1])).reshape(-1,1).astype(np.float32)
        trunc_z = ((coord[:,2] >= 0) * (coord[:,2] < output_shape[0])).reshape(-1,1).astype(np.float32)
        if combine:
            return trunc_x * trunc_y * trunc_z
        else:
            return trunc_x, trunc_y, trunc_z
    else:
        trunc_x = ((coord[:,0] >= 0) * (coord[:,0] < output_shape[1])).reshape(-1,1).astype(np.float32)
        trunc_y = ((coord[:,1] >= 0) * (coord[:,1] < output_shape[0])).reshape(-1,1).astype(np.float32)
        if combine:
            return trunc_x * trunc_y
        else:
            return trunc_x, trunc_y
