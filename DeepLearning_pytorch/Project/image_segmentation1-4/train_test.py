import os.path

from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from slide_dataset import SlideDatasets
from image_segmentation import u2net,Log_weights,T_Net
from torchvision.utils import save_image
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cup') #选择设备

weight_path = 'unet_params.pth'  #保存权重的地址
data_path = r'F:\result\dataset_segmentation1-4'  #数据地址
save_path = r'F:\result\dataset_segmentation1-4\save_dir'

if __name__ == '__main__':
    dataloader = DataLoader(SlideDatasets(data_path,4),batch_size=4,
                                      drop_last=True,
                                      shuffle=True,
                                      pin_memory=True)
    net = u2net.U2NET().to(device)

    # 判断权重是否存在
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))  # 加载权重
        print('load weight successfully')
    else:
        print('failed to load weight ')

    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(4):
        running_loss = 0
        for i, (original_image, labels) in enumerate(dataloader):

            # 放到设备上
            original_image, labels = original_image.to(device), labels.to(device)

            out_image = net(original_image)
            loss = F.binary_cross_entropy(out_image, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()  # loss累加

            if i % 50 == 0:  # 每500个batch打印损失
                print(f'{epoch}-{i+1}-loss===>{running_loss.item()}')

            if i % 2 == 0:  # 每2个eopch保存参数
                torch.save(net.state_dict(), weight_path)

            _image = original_image[0]
            _segment_image = labels[0]
            _out_image = out_image[0]

            # 拼接原图、标签、分割后的图
            img = torch.stack([_image, _segment_image, _out_image], dim = 0)
            save_image(img,f'save_path/{i}.png')

        epoch += 1

