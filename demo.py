from train import device
import os
from lapsrn import *
from PIL import Image, ImageFilter
import torchvision.transforms.functional as tf
from torchvision import transforms
import swanlab

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


def get_y(img):
    img = img.convert('YCbCr')
    img = img.getchannel(0)

    return img


def get_y_cb_cr(img):
    img_ycbcr = img.convert('YCbCr')
    y, cb, cr = img_ycbcr.split()
    return y, cb, cr


if __name__ == '__main__':
    run = swanlab.init()
    checkpoint = torch.load('ckp.pt', map_location='cuda:0')
    net = LapSrnMS(5, 5, 4)
    net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(checkpoint['state_dict'])
    net.to('cuda')

    # 分离三个通道
    y, cb, cr = get_y_cb_cr(Image.open("out_lr.png"))

    im = tf.to_tensor(y)
    im = im.unsqueeze(0)
    im = im.to('cuda:0')

    with torch.no_grad():
        out_2x, out_4x = net(im)
        out_2x[out_2x > 1] = 1
        out_4x[out_4x > 1] = 1

    out_2x = transforms.ToPILImage()(out_2x[0].cpu())
    out_4x = transforms.ToPILImage()(out_4x[0].cpu())

    cb_2x = cb.resize(out_2x.size, Image.BICUBIC)
    cr_2x = cr.resize(out_2x.size, Image.BICUBIC)

    cb_4x = cb.resize(out_4x.size, Image.BICUBIC)
    cr_4x = cr.resize(out_4x.size, Image.BICUBIC)

    # 将处理后的Y通道与原始的Cb、Cr通道合并
    out_2x = Image.merge("YCbCr", [out_2x.convert('L'), cb_2x, cr_2x])
    out_4x = Image.merge("YCbCr", [out_4x.convert('L'), cb_4x, cr_4x])

    # 将YCbCr格式的图像转换回RGB格式
    out_2x = out_2x.convert('RGB')
    out_4x = out_4x.convert('RGB')

    out_2x.save("out_2x.png", "PNG")
    out_4x.save("out_4x.png", "PNG")

    examples = []
    path = ["./out_lr.png", "./out_2x.png", "./out_4x.png"]
    caption = ["LR", "2x", "4x"]
    for i in range(3):
        print(path[i])
        examples.append(swanlab.Image(path[i], caption=caption[i]))

    run.log({"examples": examples})
    
    