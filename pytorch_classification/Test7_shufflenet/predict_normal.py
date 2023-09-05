import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from model import shufflenet_v2_x1_0


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, ], [0.229,])])

    # create model
    model = shufflenet_v2_x1_0(num_classes=25).to(device)
    # load model weights
    model_weight_path = args.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 加载类别,label标签,一行一个
    class_labels = []

    with open("label.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
    print('class_label = {}'.format(class_labels))

    # load image
    data_path = args.data_path
    assert os.path.exists(data_path), "file: '{}' dose not exist.".format(data_path)
    i = -1
    correct = 0
    for filename in os.listdir(data_path):
        i += 1
        img_path = os.path.join(data_path,filename)
        img = Image.open(img_path)
        img = img.convert('L')
        # plt.imshow(img)
    # [N, C, H, W]
        img = data_transform(img)
    # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            print('predict_cla = {}'.format(predict_cla))
            # print('class_labels[i] = {}'.format(type(class_labels[i])))
            # print('equal or not = {}'.format(str(predict_cla) == class_labels[i]))
            if str(predict_cla) == class_labels[i]:
                correct += 1

    print('class: {}   accurate : {:.4}'.format(class_indict[str(class_labels[0])], correct/(i+1)))

            # print('predict_cla = {}'.format(predict_cla))





    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)