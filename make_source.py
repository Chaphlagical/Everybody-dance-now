'''Download and extract video'''
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from pathlib import Path
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

filename="mv"
begin=800
num=300

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(0)

save_dir = Path('./data/source/'+filename)
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

'''if len(os.listdir('./data/source/'+filename+'/images'))<100:
    cap = cv2.VideoCapture(str(save_dir.joinpath(filename+'.mp4')))
    i = j = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if flag == False or i >= begin+num:
            break
        if i>=begin:
            cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(j))), frame)
            save(j)
            j+=1
        if j%100 == 0:
            print('Has generated %d picetures'%j)
        i += 1
        '''

'''Pose estimation (OpenPose)'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

openpose_dir = Path('./src/PoseEstimation/')

import sys
sys.path.append(str(openpose_dir))
sys.path.append('./src/utils')


# openpose
#from network.rtpose_vgg import gopenpose_diret_model
from evaluate.coco_eval import get_multiplier, get_outputs
from network.rtpose_vgg import get_model
# utils
from openpose_utils import remove_noise, get_pose


weight_name = './src/PoseEstimation/network/weight/pose_model.pth'

model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

'''make label images for pix2pix'''
test_img_dir = save_dir.joinpath('test_img')
test_img_dir.mkdir(exist_ok=True)
test_label_dir = save_dir.joinpath('test_label_ori')
test_label_dir.mkdir(exist_ok=True)
test_head_dir = save_dir.joinpath('test_head_ori')
test_head_dir.mkdir(exist_ok=True)

pose_cords = []

def save(idx):
    if not os.path.exists(str(test_img_dir.joinpath('{:05}.png'.format(idx)))):
        global pose_cords
        try:
            img_path = img_dir.joinpath('{:05}.png'.format(idx))
            img = cv2.imread(str(img_path))
            shape_dst = np.min(img.shape[:2])
            oh = (img.shape[0] - shape_dst) // 2
            ow = (img.shape[1] - shape_dst) // 2

            img = img[oh:oh + shape_dst, ow:ow + shape_dst]
            img = cv2.resize(img, (512, 512))
            multiplier = get_multiplier(img)
            with torch.no_grad():
                paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
            r_heatmap = np.array([remove_noise(ht)
                                  for ht in heatmap.transpose(2, 0, 1)[:-1]]) \
                .transpose(1, 2, 0)
            heatmap[:, :, :-1] = r_heatmap
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            label, cord = get_pose(param, heatmap, paf)
            index = 13
            crop_size = 25
            try:
                head_cord = cord[index]
            except:
                head_cord = pose_cords[-1]  # if there is not head point in picture, use last frame

            pose_cords.append(head_cord)
            head = img[int(head_cord[1] - crop_size): int(head_cord[1] + crop_size),
                   int(head_cord[0] - crop_size): int(head_cord[0] + crop_size), :]
            #        plt.imshow(head)
            plt.savefig(str(test_head_dir.joinpath('pose_{}.jpg'.format(idx))))
            plt.clf()
            cv2.imwrite(str(test_img_dir.joinpath('{:05}.png'.format(idx))), img)
            cv2.imwrite(str(test_label_dir.joinpath('{:05}.png'.format(idx))), label)
            if idx % 100 == 0 and idx != 0:
                pose_cords_arr = np.array(pose_cords, dtype=np.int)
                np.save(str((save_dir.joinpath('pose_source.npy'))), pose_cords_arr)
            return True
        except:
            return False
    else:
        return False







'''for idx in tqdm(range(len(os.listdir(str(img_dir))))):
    if not os.path.exists(str(test_img_dir.joinpath('{:05}.png'.format(idx)))):
        img_path = img_dir.joinpath('{:05}.png'.format(idx))
        img = cv2.imread(str(img_path))
        shape_dst = np.min(img.shape[:2])
        oh = (img.shape[0] - shape_dst) // 2
        ow = (img.shape[1] - shape_dst) // 2

        img = img[oh:oh + shape_dst, ow:ow + shape_dst]
        img = cv2.resize(img, (512, 512))
        multiplier = get_multiplier(img)
        with torch.no_grad():
            paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
        r_heatmap = np.array([remove_noise(ht)
                              for ht in heatmap.transpose(2, 0, 1)[:-1]]) \
            .transpose(1, 2, 0)
        heatmap[:, :, :-1] = r_heatmap
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        label, cord = get_pose(param, heatmap, paf)
        index = 13
        crop_size = 25
        try:
            head_cord = cord[index]
        except:
            head_cord = pose_cords[-1] # if there is not head point in picture, use last frame

        pose_cords.append(head_cord)
        head = img[int(head_cord[1] - crop_size): int(head_cord[1] + crop_size),
               int(head_cord[0] - crop_size): int(head_cord[0] + crop_size), :]
#        plt.imshow(head)
        plt.savefig(str(test_head_dir.joinpath('pose_{}.jpg'.format(idx))))
        plt.clf()
        cv2.imwrite(str(test_img_dir.joinpath('{:05}.png'.format(idx))), img)
        cv2.imwrite(str(test_label_dir.joinpath('{:05}.png'.format(idx))), label)
        if idx % 100 == 0 and idx != 0:
            pose_cords_arr = np.array(pose_cords, dtype=np.int)
            np.save(str((save_dir.joinpath('pose_source.npy'))), pose_cords_arr)
'''


if len(os.listdir('./data/source/'+filename+'/images'))<100:
    cap = cv2.VideoCapture(str(save_dir.joinpath(filename+'.mp4')))
    i = j = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if flag == False or i >= begin+num:
            break
        if i>=begin:
            cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(j))), frame)
            if save(j):
                j+=1
                print('Has generated %d picetures' % j)
        #if j%100 == 0:
         #   print('Has generated %d picetures'%j)
        i += 1


pose_cords_arr = np.array(pose_cords, dtype=np.int)
np.save(str((save_dir.joinpath('pose_source.npy'))), pose_cords_arr)
torch.cuda.empty_cache()
