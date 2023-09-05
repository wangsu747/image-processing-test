import torch
import os
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import resnet34
import numpy as np
def mian(args):
    # 设置用于评估的设备（CPU或GPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # 加载预训练的ResNet模型
    net = resnet34()
    model_weight_path = args.weights
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    net.to(device)
    # 将模型移动到所选设备
    net = net.to(device)

    # 设置数据预处理和加载数据
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "train_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    # 加载验证集或测试集数据
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),  transform=transform)
    val_num = len(val_dataset)
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    #得到class
    class_list = val_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())

    # 设置模型为评估模式
    net.eval()

    # 初始化变量以跟踪正确的预测数量和总样本数量
    correct = 0
    total = 0

    # 初始化变量以跟踪每个类别的正确预测数量和总样本数量
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    # 不进行梯度计算
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)

        for val_data in val_bar:
            val_images, val_labels = val_data
            labels = val_labels.to(device)
            outputs = net(val_images.to(device))

            predict_y = torch.max(outputs, dim=1)[1]
            print('predict_y no [1] = {}'.format(torch.max(outputs, dim=1)))
            print('')
            # 计算每个类别的准确率
            c = (predict_y == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = resnet(images)

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)

            # 更新正确的预测数量和总样本数量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算精度
    accuracy = 100 * correct / total
    print(f'Accuracy on validation/test set: {accuracy:.2f}%')