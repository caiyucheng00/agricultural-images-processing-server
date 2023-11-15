#
# @Description:采用torchvision.model进行简单的图片分类
# @Author:虚幻的元亨利贞
# @Time: 2022-11-12 16:04
#

import os
from tqdm import tqdm
import torch
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from thop import profile


class SimpleClassification:
    def __init__(self, dataset_dir):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
        self.train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                   ])

        # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
        self.test_transform = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                                                  ])
        self.dataset_dir = dataset_dir
        # 检查文件夹是否存在
        #if not os.path.exists('checkpoints'):
         #   os.mkdir('checkpoints')
        #if not os.path.exists('output'):
         #   os.mkdir('output')
        #if not os.path.exists('test_img'):
         #   os.mkdir('test_img')

    # 开始训练 输入实参dataset_dir是数据集文件夹
    def train(self, model=models.resnet18(pretrained=True), epoch=20, batch_size=1, threshold=90):
        train_path = os.path.join(self.dataset_dir, 'train')
        test_path = os.path.join(self.dataset_dir, 'val')
        # 载入训练集
        train_dataset = datasets.ImageFolder(train_path, self.train_transform)
        # 载入测试集
        test_dataset = datasets.ImageFolder(test_path, self.test_transform)
        # 各类别名称
        class_names = train_dataset.classes
        n_class = len(class_names)
        # 映射关系：类别 到 索引号
        # 映射关系：索引号 到 类别
        idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
        # 保存为本地的 npy 文件
        np.save('idx_to_labels.npy', idx_to_labels)
        np.save('labels_to_idx.npy', train_dataset.class_to_idx)

        # 训练集的数据加载器
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True
                                  )
        # 测试集的数据加载器
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False
                                 )

        # 载入预训练模型
        model.fc = nn.Linear(model.fc.in_features, n_class)
        optimizer = optim.Adam(model.parameters())
        model = model.to(self.device)
        # 交叉熵损失函数
        criterion = nn.CrossEntropyLoss()

        r = 0
        while r < threshold:
            # 遍历每个 EPOCH
            for epoch in tqdm(range(epoch)):

                model.train()

                for images, labels in train_loader:  # 获得一个 batch 的数据和标注
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    # 输入模型，执行前向预测
                    outputs = model(images)  ## outputs.shape = (1,3)
                    loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值
                    # 反向传播“三部曲”
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # 测试集测试
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in tqdm(test_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum()
                    r = 100 * correct / total

                print('测试集上的准确率为 {:.3f} %'.format(r))
            # 保存模型
            if r > threshold:
                torch.save(model, 'checkpoints/' + self.dataset_dir + '.pt')

    # 开始分类 输入实参model_name,img_name
    def classify(self,model_name, img_name, table_name, img_output_path='static/detect/img_pred.jpg'):
        #font = ImageFont.truetype('SimHei.ttf', 32)
        idx_to_labels = np.load('register_table/' + table_name, allow_pickle=True).item()
        # 导入模型
        model = torch.load(model_name, map_location='cpu')
        model = model.eval().to(self.device)
        # 加载图片
        img_pil = Image.open(img_name)
        input_img = self.test_transform(img_pil)  # 预处理
        input_img = input_img.unsqueeze(0).to(self.device)
        # 执行前向预测，得到所有类别的 logit 预测分数
        pred_logits = model(input_img)
        pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
        n = 2
        top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
        pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
        confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析出置信度

        res = ''
        draw = ImageDraw.Draw(img_pil)
        for i in range(n):
            if i == 0:
                class_name = idx_to_labels[pred_ids[i]]  # 获取类别名称
                confidence = confs[i] * 100  # 获取置信度
                #print(class_name)
                # 文字坐标，中文字符串，字体，rgba颜色
                #draw.text((50, 50), class_name, font=font, fill=(255, 0, 0, 1))
                res = class_name
        # 保存图像
        img_pil.save(img_output_path)

        # 模型参数数量
        #_, params = profile(model, inputs=(input_img,))
        #print(params)
        return res


if __name__ == '__main__':
    """
    使用教程：
    1.在test_img下放入训练图片
    2.train()调用后放入checkpoints文件夹下
    3.classify()调用后结果在output文件夹下
    """
    sc = SimpleClassification()
    sc.train()
    sc.classify()
