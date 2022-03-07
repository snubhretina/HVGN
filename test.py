import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageEnhance
from scipy import ndimage
from sklearn.metrics import auc, precision_recall_curve, roc_curve, precision_recall_fscore_support
from torch.autograd import Variable
from torchvision import datasets, models, transforms, utils
from datetime import datetime
import csv, glob
import cv2
import skimage.morphology
import segmentation_models_pytorch.base.modules as md
from model import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Deep Vessel')
parser.add_argument('--batch', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--input_path', type=str, default='')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=100, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--db', type=str, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def get_pr_auc(gt, pred):
    precision, recall, threshold = precision_recall_curve(gt, pred)
    pr_auc = auc(recall, precision)
    return precision, recall, threshold, pr_auc


def get_roc_auc(gt, pred):
    fpr, tpr, _ = roc_curve(gt, pred, pos_label=1.)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# data load in path
def load_data_path(img_path):
    cur_dir_file_list = sorted(glob.glob(img_path + "/*.png"))
    FAG_path_list = []
    FAG_img_list = []

    FAG_img_array = np.zeros([24, nY, nX], dtype=np.float32)
    FP_label_array = np.zeros([2, nY, nX], dtype=np.float32)

    for file2 in cur_dir_file_list:
        if file2.find('img_.png') != -1 or file2.find('1_origin.png') != -1:
            FP_img = [file2,
                      np.array(Image.open(file2).resize(size=(nX, nY)), dtype=np.float32)]
            FAG_img_array[1] = FP_img[1][:, :, 0]
            FAG_img_array[2] = FP_img[1][:, :, 1]
            FAG_img_array[3] = FP_img[1][:, :, 2]

    cnt = 4
    FAG_image_list = []
    for file2 in cur_dir_file_list:
        if file2.find('FAG2FP') != -1:
            FAG_image_list.append(
                np.array(Image.open(file2).resize(size=(nX, nY)).convert('L'), dtype=np.float32))
            # FAG_img_array[cnt] = np.array(Image.open(cur_dir_path + file2).resize(size=(nX, nY)), dtype=np.float32)
            cnt += 1

            if cnt == FAG_img_array.shape[0]:
                break
    mean_intensity = np.mean(FAG_image_list, axis=(1, 2))
    max_idx = np.argmax(mean_intensity)
    FAG_stacked_arr = np.zeros([20, nY, nX])

    for idx in range(len(FAG_image_list)):
        adjust_idx = 10 - (max_idx - idx)
        if adjust_idx < 0 or adjust_idx >= 20:
            continue
        FAG_stacked_arr[10 - (max_idx - idx)] = FAG_image_list[idx]
    FAG_img_array[4:] = FAG_stacked_arr.copy()

    for file2 in cur_dir_file_list:
        if file2.find('A-V_point_img') != -1 or file2.find('classification.png') != -1:
            point_data = [file2,
                          np.array(Image.open(file2).resize(size=(nX, nY)))[:, :, :3]]
            point_data[1] = np.round(point_data[1].astype(np.float32) / 255)

    label = np.zeros(point_data[1].shape)
    artery = point_data[1][:, :, 0]
    vein = point_data[1][:, :, 2]
    label[artery != 0, 0] = 1
    label[vein != 0, 2] = 2

    FAG_img_array[0] = cv2.bitwise_or((artery).astype(np.float32),
                                      (vein).astype(np.float32))*255

    FP_label_array[0] = label[:, :, 0]
    FP_label_array[1] = label[:, :, 2]

    return FAG_img_array, FP_label_array


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CUDA_VISIBLE_DEVICES=3
max_iter = 50000
batch_size = args.batch
log_test = 200

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

# DB_root_path = '../result_20210518_interpolation_to_12_balancing_vessel/'
# DB_root_path = '../result_20210513_extrapolation/'
DB_root_path = '../result_20210518/'
DB_data_path = DB_root_path + 'data/'

nY = 1024
nX = 1536
train_data_path = []
test_data_path = []

fundus_count = 0
bn_count = 0

patient_dict = {}

for set_name in sorted(os.listdir(DB_root_path)):
    if set_name.split("/")[-1].split(".").__len__() > 1:
        continue
    set_path = DB_root_path + set_name + '/'
    file_list = sorted(os.listdir(set_path))
    print(file_list)
    print(file_list.__len__())
    for DB_data_idx, file in enumerate(file_list):
        patient_number = file.split("/")[-1].split("_")[0]
        if patient_dict.keys().__len__() == 0:
            patient_dict[patient_number] = 1
        else:
            if patient_number in patient_dict.keys():
                patient_dict[patient_number] += 1
            else:
                patient_dict[patient_number] = 1

        if DB_data_idx < int(sorted(os.listdir(DB_data_path)).__len__() * 0.8):
            train_data_path.append(set_path + file + '/')
        else:
            test_data_path.append(set_path + file + '/')

print(train_data_path.__len__(), test_data_path.__len__(), 'lr : {}'.format(args.lr))


def random_perturbation(imgs):
    for i in range(imgs.shape[0]):
        im = Image.fromarray(imgs[i, ...].astype(np.uint8))
        en = ImageEnhance.Color(im)
        im = en.enhance(np.random.uniform(0.8, 1.2))
        imgs[i, ...] = np.asarray(im).astype(np.float32)
    return imgs


def load_te_img(idx):
    CNN_data = torch.FloatTensor(1, 24, nY, nX)
    CNN_label = torch.LongTensor(1, 2, nY, nX)
    path = test_data_path[idx]
    test_data = load_data_path(path)
    numpy_data = test_data[0].copy()
    for i in range(1, 24):
        numpy_data[i] -= np.mean(numpy_data[i, numpy_data[0] != 0])
    CNN_data[0] = torch.from_numpy(numpy_data)
    CNN_label[0] = torch.from_numpy(test_data[1][:])

    return CNN_data, CNN_label

def test():
    model.eval()
    h, w = 1024, 1536
    t = 0
    # iter = 5
    iter = len(test_data_path)

    size_count = 0
    corr_count = 0
    pred_list = []
    gt_list = []
    with torch.no_grad():
        for i in range(iter):

            # CNN_data, CNN_label, edge_index, label
            cnn_data, cnn_label = load_te_img(i)
            cnn_data, cnn_label = cnn_data.cuda("cuda:0"), \
                                  cnn_label.cuda("cuda:0")
            st = datetime.now()
            outputs = model(cnn_data, cnn_label)
            et = datetime.now()
            t += (et-st).total_seconds()
            gnn_label_A = cnn_label[0, 0, cnn_data[0, 0] != 0]
            mt_gnn_label_A = torch.zeros([gnn_label_A.size(0), 2])
            mt_gnn_label_A[gnn_label_A == 0, 0] = 1
            mt_gnn_label_A[gnn_label_A == 1, 1] = 1

            gnn_label_V = cnn_label[0, 1, cnn_data[0, 0] != 0]
            mt_gnn_label_V = torch.zeros([gnn_label_V.size(0), 2])
            mt_gnn_label_V[gnn_label_V == 0, 0] = 1
            mt_gnn_label_V[gnn_label_V == 2, 1] = 1

            mt_gnn_label_A = mt_gnn_label_A.cuda("cuda:0")
            mt_gnn_label_V = mt_gnn_label_V.cuda("cuda:0")


            for output in outputs:
                pred = output.view(-1).data.cpu().numpy().copy()
                pred_list = pred_list + [v for v in pred]
            for mt_gnn_label in [mt_gnn_label_A, mt_gnn_label_V]:
                label_numpy = mt_gnn_label.view(-1).data.cpu().numpy().copy()
                gt_list = gt_list + [v for v in label_numpy]
            nonzero_axis = np.nonzero(cnn_data[0][0])
            canvas_output = np.zeros([h, w, 3], dtype=np.ubyte)
            for idx, gnn_label_idx in enumerate([gnn_label_A, gnn_label_V]):
                _, pred = F.softmax(outputs[idx]).max(dim=1)
                for j in range(gnn_label_idx.size(0)):
                    if idx == 0:
                        if pred[j].item() == 1:
                            canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], 0] = 255
                        else:
                            canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], 2] = 255
                    else:
                        if pred[j].item() == 1:
                            canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], 2] = 255
                        else:
                            canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], 0] = 255
            output_save_path = save_dir_path + '%05d.png' % (i)
            Image.fromarray(canvas_output).save(output_save_path)

    pred_arr = np.array(pred_list).flatten()
    gt_arr = np.array(gt_list).flatten()

    precision, recall, threshold, pr_auc_score = get_pr_auc(gt_arr, pred_arr)
    fpr, tpr, roc_auc_score = get_roc_auc(gt_arr, pred_arr)

    all_f1 = 2. * precision * recall / (precision + recall)
    best_f1 = np.nanmax(all_f1)
    index = np.nanargmax(all_f1)
    best_f1_threshold = threshold[index]
    binary_flat = (pred_arr >= best_f1_threshold).astype(np.float32)
    acc = (gt_arr == binary_flat).sum() / float(pred_arr.shape[0])

    tp = np.bitwise_and((gt_arr == 1).astype(np.ubyte), (binary_flat == 1).astype(np.ubyte)).sum()
    tn = np.bitwise_and((gt_arr == 0).astype(np.ubyte), (binary_flat == 0).astype(np.ubyte)).sum()
    fp = np.bitwise_and((gt_arr == 0).astype(np.ubyte), (binary_flat == 1).astype(np.ubyte)).sum()
    fn = np.bitwise_and((gt_arr == 1).astype(np.ubyte), (binary_flat == 0).astype(np.ubyte)).sum()
    se = tp / float(tp + fn)
    sp = tn / float(fp + tn)

    score = [pr_auc_score, roc_auc_score, best_f1, best_f1_threshold, acc, se, sp]

    return score

model = Net()
db_path = args.db
db_count = db_path.split("/")[-2]
print(db_count)
model.CNN.load_state_dict(torch.load(db_path + '_cnn.pth.tar'))
model.GUnet_A.load_state_dict(
    torch.load(db_path + '_artery.pth.tar'))
model.GUnet_V.load_state_dict(
    torch.load(db_path + '_vein.pth.tar'))
save_dir_path = "./result_img_db_{}/".format(db_count)
os.makedirs(save_dir_path, exist_ok=True)

pr, roc, bestf1, bestf1_threshold, acc, se, sp = test()
f = open(save_dir_path + 'all_test_result.csv', 'w')
csv_file = csv.writer(f)
csv_file.writerow(
    ['PR', 'ROC', 'F1', 'thresh', 'ACC', 'SE', 'SP'])
csv_file.writerow([pr, roc, bestf1, bestf1_threshold, acc, se, sp])

f.close()


