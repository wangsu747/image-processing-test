import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import shufflenet_v2_x1_0


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data_transform = transforms.Compose(
    #     [transforms.Resize(256),
    #      transforms.CenterCrop(224),
    #      transforms.ToTensor(),
    #      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, ], [0.229, ])])

    # 加载类别,label标签,一行一个
    class_labels = []
    with open("label.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # load image
    img_path = args.data_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    accuracies = [0] * len(class_labels)
    class_counts = [0] * len(class_labels)



    # create model
    model = shufflenet_v2_x1_0(num_classes=25).to(device)
    # load model weights
    model_weight_path = args.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    #遍历数据
    for filename in os.listdir(img_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(img_path, filename) #每个图片的绝对路径
            img = Image.open(image_path)
            plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
        # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy() #第几个数字类
            print("class: {:10}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                  predict[predict_cla].numpy()))



    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_classes', type=int, default=25)
    # parser.add_argument('--epochs', type=int, default=30)
    # parser.add_argument('--batch-size', type=int, default=2)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--lrf', type=float, default=0.1)


    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # shufflenetv2_x1.0 官方权重下载地址
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    # parser.add_argument('--weights', type=str, default='./shufflenetv2_x1.pth',
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)