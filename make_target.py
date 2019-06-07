import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

filename="demo"
begin=0
num=1000

openpose_dir = Path('./src/PoseEstimation/')


save_dir = Path('./data/target/'+filename)
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)


'''if len(os.listdir('./data/target/'+filename+'/images'))<100:
    cap = cv2.VideoCapture(str(save_dir.joinpath(filename + '.mp4')))
    i = j = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if flag == False or i >= begin + num:
            break
        if i >= begin:
            cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(j))), frame)
            j += 1
        if j % 100 == 0:
            print('Has generated %d picetures' % j)
        i += 1
        '''

import sys
sys.path.append(str(openpose_dir))
sys.path.append('./src/utils')
# openpose
from network.rtpose_vgg import get_model
from evaluate.coco_eval import get_multiplier, get_outputs

# utils
from openpose_utils import remove_noise, get_pose

weight_name = './src/PoseEstimation/network/weight/pose_model.pth'
print('load model...')
model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()
pass

save_dir = Path('./data/target/'+filename)
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)


'''make label images for pix2pix'''
train_dir = save_dir.joinpath('train')
train_dir.mkdir(exist_ok=True)

train_img_dir = train_dir.joinpath('train_img')
train_img_dir.mkdir(exist_ok=True)
train_label_dir = train_dir.joinpath('train_label')
train_label_dir.mkdir(exist_ok=True)
train_head_dir = train_dir.joinpath('head_img')
train_head_dir.mkdir(exist_ok=True)

pose_cords = []

def save(idx):
    global pose_cords
    if not os.path.exists(str(train_img_dir.joinpath('{:05}.png'.format(idx)))):
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
                                  for ht in heatmap.transpose(2, 0, 1)[:-1]]).transpose(1, 2, 0)
            heatmap[:, :, :-1] = r_heatmap
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            # TODO get_pose
            label, cord = get_pose(param, heatmap, paf)
            index = 13
            crop_size = 25
            try:
                head_cord = cord[index]
            except:
                try:
                    head_cord = pose_cords[-1]  # if there is not head point in picture, use last frame
                except:
                    head_cord = None

            pose_cords.append(head_cord)
            try:
                head = img[int(head_cord[1] - crop_size): int(head_cord[1] + crop_size),
                       int(head_cord[0] - crop_size): int(head_cord[0] + crop_size), :]
            except:
                pass
            #    plt.imshow(head)
            plt.savefig(str(train_head_dir.joinpath('pose_{}.jpg'.format(idx))))
            plt.clf()
            cv2.imwrite(str(train_img_dir.joinpath('{:05}.png'.format(idx))), img)
            cv2.imwrite(str(train_label_dir.joinpath('{:05}.png'.format(idx))), label)
            return True
        except:
            return False

    else:
        return False


if len(os.listdir('./data/target/'+filename+'/images'))<100:
    cap = cv2.VideoCapture(str(save_dir.joinpath(filename + '.mp4')))
    i = j = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if flag == False or i >= begin + num:
            break
        if i >= begin:
            cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(j))), frame)
            cv2.imshow("aaa",frame)
            cv2.waitKey(1)
            if save(j):
                j += 1
                print('Has generated %d picetures' % j)
            else:
                print("save failed")
        i += 1


'''
for idx in tqdm(range(len(os.listdir(str(img_dir))))):
    if not os.path.exists(str(train_img_dir.joinpath('{:05}.png'.format(idx)))):
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
                              for ht in heatmap.transpose(2, 0, 1)[:-1]]).transpose(1, 2, 0)
        heatmap[:, :, :-1] = r_heatmap
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        #TODO get_pose
        label, cord = get_pose(param, heatmap, paf)
        index = 13
        crop_size = 25
        try:
            head_cord = cord[index]
        except:
            try:
                head_cord = pose_cords[-1] # if there is not head point in picture, use last frame
            except:
                head_cord=None

        pose_cords.append(head_cord)
        try:
            head = img[int(head_cord[1] - crop_size): int(head_cord[1] + crop_size),
               int(head_cord[0] - crop_size): int(head_cord[0] + crop_size), :]
        except:
            pass
    #    plt.imshow(head)
        plt.savefig(str(train_head_dir.joinpath('pose_{}.jpg'.format(idx))))
        plt.clf()
        cv2.imwrite(str(train_img_dir.joinpath('{:05}.png'.format(idx))), img)
        cv2.imwrite(str(train_label_dir.joinpath('{:05}.png'.format(idx))), label)'''

pose_cords = np.array(pose_cords, dtype=np.int)
np.save(str((save_dir.joinpath('pose.npy'))), pose_cords)
torch.cuda.empty_cache()
