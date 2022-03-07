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
parser = argparse.ArgumentParser(description='PyTorch Combining Fundus Images and Fluorescein Angiography for Artery/Vein Classification Using the Hierarchical Vessel Graph Network')
parser.add_argument('--batch', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--input_path', type=str, default='')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=100, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train-continue', type=bool, default=False,
                    help='load model for continuing train')
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


def load_tr_img():
    randp = np.random.random_integers(0, len(train_data_path) - 1, 1)

    path = train_data_path[randp[0]]
    train_data = load_data_path(path)

    CNN_data = torch.FloatTensor(1, 24, nY, nX)
    CNN_label = torch.LongTensor(1, 2, nY, nX)

    numpy_data = train_data[0].copy()
    numpy_label = train_data[1].copy()

    if np.random.random_sample(1) > 0.5:
        numpy_data = numpy_data[:, :, ::-1].copy()
        numpy_label = numpy_label[:, :, ::-1].copy()

    binary_img = (numpy_data[0:1]).astype(np.float32)
    FP_img = numpy_data[1:]

    FP_img += np.random.uniform(-0.3, 0.3) * 255
    FP_img = np.clip(FP_img, 0, 255)

    mmR = np.mean(FP_img[0, :, :])
    mmG = np.mean(FP_img[1, :, :])
    mmB = np.mean(FP_img[2, :, :])
    rand_contrast = np.random.uniform(0.5,
                                      1.5, 1)
    FP_img[0] = (FP_img[0] - mmR) * rand_contrast[0] + mmR
    FP_img[1] = (FP_img[1] - mmG) * rand_contrast[0] + mmG
    FP_img[2] = (FP_img[2] - mmB) * rand_contrast[0] + mmB
    FP_img = np.clip(FP_img, 0, 255)

    numpy_data[0] = binary_img
    numpy_data[1:] = FP_img

    angle = np.random.randint(-30, 30, 1)
    for j in range(24):
        numpy_data[j] = Image.fromarray(numpy_data[j]).rotate(angle)

    for j in range(2):
        numpy_label[j] = np.array(Image.fromarray(numpy_label[j]).rotate(angle), dtype=np.float32)

    for j in range(1, 24):
        numpy_data[j] -= np.mean(numpy_data[j, numpy_data[0] != 0])

    CNN_data[0] = torch.from_numpy(numpy_data)
    CNN_label[0] = torch.from_numpy(numpy_label)

    return CNN_data, CNN_label

def train(iter, t, train_loss_list):
    st = datetime.now()
    model.train()

    # CNN_data, CNN_label
    cnn_data, cnn_label = load_tr_img()
    cnn_data, cnn_label = cnn_data.cuda("cuda:0"), \
                          cnn_label.cuda("cuda:0")

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

    optimizer.zero_grad()
    
    output1, output2 = model(cnn_data, cnn_label)

    loss_A = criterion(output1, mt_gnn_label_A)
    loss_V = criterion(output2, mt_gnn_label_V)

    loss_a_cpu = loss_A.item()
    loss_v_cpu = loss_V.item()
    loss = (loss_A + loss_V)
    loss.backward()

    optimizer.step()
    scheduler.step()
    et = datetime.now()
    t += (et - st).total_seconds()

    train_loss_list.append(loss_a_cpu + loss_v_cpu)

    print('acc: %0.4f' % (loss))

    return t, train_loss_list, loss


def test(test_loss_list):
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
            st = datetime.now()
            # CNN_data, CNN_label, edge_index, label
            cnn_data, cnn_label = load_te_img(i)
            cnn_data, cnn_label = cnn_data.cuda("cuda:0"), \
                                  cnn_label.cuda("cuda:0")

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
            outputs = model(cnn_data, cnn_label)

            loss_A = criterion(outputs[0], mt_gnn_label_A)
            loss_V = criterion(outputs[1], mt_gnn_label_V)

            lossA = loss_A.item()
            lossV = loss_V.item()
            loss = (lossA + lossV)

            test_loss_list.append(loss)

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
                    if pred[j].item() == 1:
                        canvas_output[nonzero_axis[j][0], nonzero_axis[j][1], idx * 2] = 255
            output_save_path = save_dir_path + "2_AV_mask/" + '%05d.png' % (i)
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

    return test_loss_list, iter, t, score


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CUDA_VISIBLE_DEVICES=3
max_iter = 50000
batch_size = args.batch
log_test = 200

DB_root_path = '../result_20210518/'
DB_data_path = DB_root_path + 'data/'

nY = 1024
nX = 1536
train_data_path = []
test_data_path = []

fundus_count = 0
bn_count = 0

for set_name in sorted(os.listdir(DB_root_path)):
    if set_name.split("/")[-1].split(".").__len__() > 1:
        continue
    set_path = DB_root_path + set_name + '/'
    file_list = sorted(os.listdir(set_path))
    print(file_list)
    print(file_list.__len__())
    for DB_data_idx, file in enumerate(file_list):

        if DB_data_idx < int(sorted(os.listdir(DB_data_path)).__len__() * 0.8):
            train_data_path.append(set_path + file + '/')
        else:
            test_data_path.append(set_path + file + '/')

print(train_data_path.__len__(), test_data_path.__len__(), 'lr : {}'.format(args.lr))

# model2 = AV_net(VGG16).to("cuda:0")
model = Net()
train_t = 0
mean_train_loss_list = []
mean_test_loss_list = []
train_loss_list = []
test_loss_list = []
test_acc_list = []
test_pr_list = []
test_roc_list = []
test_f1_list = []
test_se_list = []
test_sp_list = []
all_test_scores = []


optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
criterion = nn.BCELoss().cuda()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

all_loss = []
save_dir_path = 'DB_AV_model_save_{}_lr_{}_0109/'.format(DB_root_path.split("/")[-2], str(args.lr))
if os.path.exists(save_dir_path) == False:
    os.mkdir(save_dir_path)
if os.path.exists(save_dir_path + "2_AV_mask") == False:
    os.mkdir(save_dir_path + "2_AV_mask")
for iter in range(max_iter + 1):

    cur_lr = optimizer.param_groups[0]['lr']

    # param_group['lr']
    train_t, train_loss_list, cur_train_loss = train(iter, train_t, train_loss_list)

    all_loss.append(cur_train_loss)
    if (iter + 1) % args.log_interval == 0:
        print('Train iter: %d [%d/%d (%.0f)] Loss: %.4f, time per frame: %.4f, total time: %.4f(bs:%d)' \
              % (max_iter, (iter + 1), max_iter,
                 100. * (iter + 1) / max_iter, cur_train_loss, train_t / float(args.log_interval) / float(batch_size),
                 train_t, batch_size))
        train_t = 0

    if (iter + 1) % log_test == 0:
        print('\ncur dir: %s, rl: %e' % (os.getcwd(), cur_lr))
        test_loss_list, test_iter, test_t, score = test(test_loss_list)

        print('\nTest set:time per frame: %.4f, total time: %.4f(%d)\n' % (
            test_t / float(test_iter * batch_size), test_t, test_iter * batch_size))

        mean_train_loss_list.append(np.mean(train_loss_list))
        mean_test_loss_list.append(np.mean(test_loss_list))
        test_pr_list.append(score[0])
        test_roc_list.append(score[1])
        test_f1_list.append(score[2])
        test_acc_list.append(score[4])
        test_se_list.append(score[5])
        test_sp_list.append(score[6])
        all_test_scores.append(score)

        cnn_save_path = save_dir_path + 'model_%d_iter_loss_%.4f_acc_%.4f_cnn.pth.tar' % (
            iter + 1, mean_test_loss_list[-1], test_acc_list[-1])
        A_save_path = save_dir_path + 'model_%d_iter_loss_%.4f_acc_%.4f_artery.pth.tar' % (
            iter + 1, mean_test_loss_list[-1], test_acc_list[-1])
        V_save_path = save_dir_path + 'model_%d_iter_loss_%.4f_acc_%.4f_vein.pth.tar' % (
            iter + 1, mean_test_loss_list[-1], test_acc_list[-1])
        torch.save(model.CNN.state_dict(), cnn_save_path)
        torch.save(model.GUnet_A.state_dict(), A_save_path)
        torch.save(model.GUnet_V.state_dict(), V_save_path)

        # draw the result
        cnt = len(mean_train_loss_list)
        lw = 1
        x_range = range(log_test, log_test * (cnt + 1), log_test)
        fig = plt.figure(figsize=(10, 8))
        plt.plot(x_range, np.array(mean_train_loss_list), 'g', lw=lw, label='train loss')
        plt.plot(x_range, np.array(mean_test_loss_list), 'b', lw=lw, label='test loss')
        plt.grid(True)
        plt.title('train-test loss')
        plt.legend(loc="upper right")
        fig.savefig(save_dir_path + '1_loss.png')
        plt.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(x_range, np.array(test_roc_list), 'g', lw=lw, label='ROC')
        plt.plot(x_range, np.array(test_pr_list), 'b', lw=lw, label='PR')
        plt.grid(True)
        plt.title('score')
        plt.legend(loc="lower right")
        fig.savefig(save_dir_path + '2_PR_ROC.png')
        plt.close()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(x_range, np.array(test_acc_list), 'g', lw=lw, label='ACC')
        plt.plot(x_range, np.array(test_f1_list), 'b', lw=lw, label='F1')
        plt.plot(x_range, np.array(test_se_list), lw=lw, label='SE')
        plt.plot(x_range, np.array(test_sp_list), lw=lw, label='SP')
        plt.grid(True)
        plt.title('score')
        plt.legend(loc="lower right")
        fig.savefig(save_dir_path + '3_ACC_F1_SE_SP.png')
        plt.close()

        f = open(save_dir_path + 'all_test_result.csv', 'w')
        csv_file = csv.writer(f)
        csv_file.writerow(
            ['iter', 'train_loss', 'test_loss', 'PR', 'ROC', 'F1', 'thresh', 'ACC', 'SE', 'SP', 'in_vessel_max_acc',
             'se', 'sp', 'precison', 'recall'])
        for i in range(mean_train_loss_list.__len__()):
            csv_file.writerow([x_range[i], mean_train_loss_list[i], mean_test_loss_list[i]] + all_test_scores[i])
        f.close()

        train_loss_list = []
        test_loss_list = []


