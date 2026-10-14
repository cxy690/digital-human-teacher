import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
# from mmpose.apis import inference_topdown, init_model
# from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm

# initialize the mmpose model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# config_file = './avatars/musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
# checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
# model = init_model(config_file, checkpoint_file, device=device)

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)

# ---- MediaPipe FaceLandmarker 替代 face_recognition(dlib)，重建 68 点等价数组 ----
# 槽位约定(与下方 bbox 微调逻辑兼容)：
#   [28]=mp168(鼻梁/眉心) [29]=mp1(鼻尖上缘) [30]=mp2(鼻底)
#   槽位 0-35 填 FACE_OVAL 轮廓点，覆盖全脸 min(x)/max(x)/max(y) 统计
_face_landmarker = None
_FACE_OVAL_IDS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                  379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                  234, 127, 162, 21, 54, 103, 67, 109]

def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        # 路径：digital_human/musetalk/utils/ → 上三级到项目根(data 在项目根下)
        model_path = os.environ.get(
            "FACE_LANDMARKER_MODEL",
            os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "..", "..", "..", "data", "models", "face_landmarker.task")))
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1,
            running_mode=vision.RunningMode.IMAGE)
        _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker

def mediapipe_face_landmarks(img_rgb):
    """mediapipe FaceLandmarker 检测人脸并重建 68 点等价数组，未检测到人脸返回 None"""
    import mediapipe as mp
    h, w = img_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
    result = _get_face_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.int32)
    face_land_mark = np.zeros((68, 2), dtype=np.int32)
    # 槽位 0-67 全部填真实坐标（否则 0 值会污染下游整列 min/max 统计）
    oval_pts = pts[_FACE_OVAL_IDS]
    face_land_mark[:len(_FACE_OVAL_IDS)] = oval_pts
    for k in range(len(_FACE_OVAL_IDS), 68):
        face_land_mark[k] = oval_pts[k % len(_FACE_OVAL_IDS)]
    face_land_mark[28] = pts[168]
    face_land_mark[29] = pts[1]
    face_land_mark[30] = pts[2]
    return face_land_mark

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

# def get_bbox_range(img_list,upperbondrange =0):
#     frames = read_imgs(img_list)
#     batch_size_fa = 1
#     batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
#     coords_list = []
#     landmarks = []
#     if upperbondrange != 0:
#         print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
#     else:
#         print('get key_landmark and face bounding boxes with the default value')
#     average_range_minus = []
#     average_range_plus = []
#     for fb in tqdm(batches):
#         results = inference_topdown(model, np.asarray(fb)[0])
#         results = merge_data_samples(results)
#         keypoints = results.pred_instances.keypoints
#         face_land_mark= keypoints[0][23:91]
#         face_land_mark = face_land_mark.astype(np.int32)
        
#         # get bounding boxes by face detetion
#         bbox = fa.get_detections_for_batch(np.asarray(fb))
        
#         # adjust the bounding box refer to landmark
#         # Add the bounding box to a tuple and append it to the coordinates list
#         for j, f in enumerate(bbox):
#             if f is None: # no face in the image
#                 coords_list += [coord_placeholder]
#                 continue
            
#             half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
#             range_minus = (face_land_mark[30]- face_land_mark[29])[1]
#             range_plus = (face_land_mark[29]- face_land_mark[28])[1]
#             average_range_minus.append(range_minus)
#             average_range_plus.append(range_plus)
#             if upperbondrange != 0:
#                 half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）

#     text_range=f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
#     return text_range
    

def get_landmark_and_bbox(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        # results = inference_topdown(model, np.asarray(fb)[0])
        # results = merge_data_samples(results)
        # keypoints = results.pred_instances.keypoints
        # face_land_mark= keypoints[0][23:91]
        # face_land_mark = face_land_mark.astype(np.int32)
        img_rgb = cv2.cvtColor(np.asarray(fb)[0], cv2.COLOR_BGR2RGB)
        _flm = mediapipe_face_landmarks(img_rgb)
        if _flm is not None:
            face_land_mark = _flm
        else:
            face_land_mark = np.zeros((68, 2), dtype=np.int32)
        
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
            min_upper_bond = 0
            upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
            
            f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
            x1, y1, x2, y2 = f_landmark
            
            if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w,h = f[2]-f[0], f[3]-f[1]
                print("error bbox:",f)
            else:
                coords_list += [f_landmark]
    
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list,frames
    

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
        
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
