import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils
import segmentation_models_pytorch.base.modules as md
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import numpy as np
from PIL import Image
from datetime import datetime

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SSANet3(nn.Module):
    def __init__(self, pretraind_model):
        super(SSANet3, self).__init__()
        self.scale_factor = 2
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 1 , 3, bias=False)
        self.conv1.load_state_dict(pretraind_model.conv1.state_dict().copy())

        self.bn1 = nn.BatchNorm2d(64)
        self.bn1.load_state_dict(pretraind_model.bn1.state_dict().copy())
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = self._make_layer(BasicBlock, 64, 3, stride=2)
        # self.conv2.load_state_dict(pretraind_model.layer1.state_dict().copy())
        self.conv3 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.conv3.load_state_dict(pretraind_model.layer2.state_dict().copy())
        self.conv4 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.conv4.load_state_dict(pretraind_model.layer3.state_dict().copy())
        self.conv5 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.conv5.load_state_dict(pretraind_model.layer4.state_dict().copy())

        self.sp1 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.sp2 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.sp3 = nn.Sequential(
            nn.Conv2d(128, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.sp4 = nn.Sequential(
            nn.Conv2d(256, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.sp5 = nn.Sequential(
            nn.Conv2d(512, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True))



        # self.output = self.make_infer(1, 1+5 * 16)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def make_infer(self, n_infer, n_in_feat):
        infer_layers = []
        for i in range(n_infer - 1):
            if i == 0:
                conv = nn.Sequential(
                    nn.Conv2d(n_in_feat, 16, 3, 1, 1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(16, 16, 3, 1, 1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(True)
                )
            infer_layers.append(conv)

        if n_infer == 1:
            infer_layers.append(nn.Sequential(nn.Conv2d(n_in_feat, 1, 1)))
        else:
            infer_layers.append(nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(True)))
            infer_layers.append(nn.Sequential(nn.Conv2d(16, 1, 1)))

        return nn.Sequential(*infer_layers)

    def forward(self, x):
        origin_size = (x.size(2), x.size(3))
        fixed_down_size = (x.size(2)//self.scale_factor, x.size(3)//self.scale_factor)
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        sp1 = self.sp1(c1)
        c1 = F.interpolate(c1, size=fixed_down_size, mode='bilinear', align_corners=True)

        c2 = self.conv2(c1)
        sp2 = F.interpolate(self.sp2(c2), size=origin_size, mode='bilinear', align_corners=True)
        # c2 = F.interpolate(c2, size=fixed_down_size, mode='bilinear', align_corners=True)

        c3 = self.conv3(c2)
        sp3 = F.interpolate(self.sp3(c3), size=origin_size, mode='bilinear', align_corners=True)
        # c3 = F.interpolate(c3, size=fixed_down_size, mode='bilinear', align_corners=True)

        c4 = self.conv4(c3)
        sp4 = F.interpolate(self.sp4(c4), size=origin_size, mode='bilinear', align_corners=True)

        c5 = self.conv5(c4)
        sp5 = F.interpolate(self.sp5(c5), size=origin_size, mode='bilinear', align_corners=True)

        Image.fromarray((sp1.cpu().data[0].numpy().mean(axis=0) * 255).astype(np.uint8)).save("tmp1.png")
        Image.fromarray((sp2.cpu().data[0].numpy().mean(axis=0) * 255).astype(np.uint8)).save("tmp2.png")
        Image.fromarray((sp3.cpu().data[0].numpy().mean(axis=0) * 255).astype(np.uint8)).save("tmp3.png")
        Image.fromarray((sp4.cpu().data[0].numpy().mean(axis=0) * 255).astype(np.uint8)).save("tmp4.png")
        Image.fromarray((sp5.cpu().data[0].numpy().mean(axis=0) * 255).astype(np.uint8)).save("tmp5.png")

        return torch.cat([sp1, sp2, sp3, sp4, sp5], 1)

def get_back_bone_model():
    model = SSANet3(models.resnet34(pretrained=True))
    return model

class FENet(nn.Module):
    def __init__(self):
        super(FENet, self).__init__()
        self.FENet_Fundus = get_back_bone_model().to("cuda:0")
        self.FENet_FA = get_back_bone_model().to("cuda:1")
        self.FENet_FA.conv1 = nn.Conv2d(20, 64, 7, 1, 3, bias=False).to("cuda:1")

    def forward(self, x):
        mask = x[:, :1]
        out_fundus = self.FENet_Fundus(x[:, 1:4])
        out_fa = self.FENet_FA(x[:, 4:].to("cuda:1"))
        output_cat = torch.cat([mask, out_fundus, out_fa.to("cuda:0")], 1)
        # output_a = self.output_a(output_cat)
        # output_v = self.output_v(output_cat)
        return output_cat

class GUNet(torch.nn.Module):
    def __init__(self):
        super(GUNet, self).__init__()
        self.GUnet = gnn.GraphUNet(80 * 2+1, 32, 2, 3)
        # self.GUnet = gnn.GraphUNet(3, 32, 2, 4)

    def forward(self, x, edge_index):
        x = self.GUnet(x, edge_index)
        return x


class Net(torch.nn.Module):
    def __init__(self, train_continue=True):
        super(Net, self).__init__()
        self.GUnet_A = GUNet().to("cuda:0")
        self.GUnet_V = GUNet().to("cuda:0")
        self.CNN = FENet()
        self.pretrained_model = torch.load('./CNN_best_model/model_12000_iter_loss_20655.9329_acc_0.9300.pth.tar')

    def get_edge_inform(self, data, search_range):
        cur_mask = data[0][0].clone()
        idx_mask = (cur_mask - 1).long()
        # idx_mask = torch.repeat_interleave(torch.unsqueeze(cur_mask, 2),2,2).long()
        vessel_idx = torch.nonzero(cur_mask, as_tuple=True)
        # cur_idx_mask_idx = torch.nonzero(cur_mask, as_tuple=False)
        idx_mask[vessel_idx] = torch.arange(0, len(vessel_idx[0]), 1).long().to("cuda:0")
        xx, yy = np.meshgrid(np.linspace(-(search_range // 2), search_range // 2, search_range),
                             np.linspace(-(search_range // 2), search_range // 2, search_range))
        xx = np.array(xx, dtype=np.int).reshape([-1])
        yy = np.array(yy, dtype=np.int).reshape([-1])
        edge_idx1 = torch.tensor([]).long().to("cuda:0")
        edge_idx2 = torch.tensor([]).long().to("cuda:0")
        for x, y in zip(xx, yy):
            cur_shift = torch.roll(cur_mask, (y, x), dims=(0, 1))
            cur_idx_mask_shift = torch.roll(idx_mask, (y, x), dims=(0, 1))
            conn_mask = (cur_mask - cur_shift) == 0
            cur_shift_idx = torch.nonzero(conn_mask[vessel_idx], as_tuple=True)
            cur_shift_pts = torch.nonzero(conn_mask[vessel_idx], as_tuple=False)
            cur_idx_mask_idx = cur_idx_mask_shift[vessel_idx][cur_shift_idx]
            edge_idx1 = torch.cat([edge_idx1, cur_shift_pts.view(-1)], 0)
            edge_idx2 = torch.cat([edge_idx2, cur_idx_mask_idx.view(-1)], 0)
        edge_index = torch.cat([torch.unsqueeze(edge_idx1, 0), torch.unsqueeze(edge_idx2, 0)], 0).long()

        return edge_index

    def forward(self, cnn_data, mask):
        feature = self.CNN(cnn_data)

        graph_feat = torch.masked_select(feature.cpu(), (cnn_data[0][0] > 0).cpu()).view([80 * 2 + 1, -1])

        edge_index_data = self.get_edge_inform(cnn_data.to("cuda:0"), 5).to("cuda:0")
        # edge_index_data = self.get_one_top_edge_inform(cnn_data)

        x_data = graph_feat.transpose(1, 0).clone().to("cuda:0")
        gnn_data = Data(x=x_data, edge_index=edge_index_data)

        # x = self.GCN(gnn_data)
        x_A = self.GUnet_A(gnn_data.x, gnn_data.edge_index)
        x_V = self.GUnet_V(gnn_data.x, gnn_data.edge_index)

        return F.sigmoid(x_A), F.sigmoid(x_V)