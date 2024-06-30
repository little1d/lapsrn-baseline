from torch.utils import data
import torch.optim as optim
from SRdataset import SRdataset
from lapsrn import *
import shutil
import os
import torch
import swanlab


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=100):
    """Decay learning rate by a factor of 2 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr

# CUDA for PyTorch
# 设置CUDA_VISIBLE_DEVICES环境变量为"1,2,3"，这样PyTorch就只能看到这三个GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# 检查CUDA是否可用（这将根据CUDA_VISIBLE_DEVICES的设置返回True，如果设置了至少一个可用的GPU）
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


if __name__ == '__main__':
    run = swanlab.init(
        experiment_name="MSLapSRN-Super-Resolution",
        description="PyTorch LapSRN implementation with weight sharing and skip connections (MSLapSRN)",
        config={
            "init_lr": 0.001,
            "lr_decay_steps": 100,
            "lr_decay_rate": 0.5,
            "epochs": 200,
            "batch_size": 64,
            "patch_size": 128
        },
        logdir="swanlog"
    )

    max_epochs = 1000

    # Generators
    training_set = SRdataset("train")
    training_generator = data.DataLoader(training_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    validation_set = SRdataset("validation")
    validation_generator = data.DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)
    print(training_generator, validation_generator)
    print(len(training_generator))
    print(len(validation_generator))

    net = LapSrnMS(5, 5, 4)

    if use_cuda:
        net = torch.nn.DataParallel(net)  # 包装模型以支持多GPU
        net.to(device)

    criterion = CharbonnierLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    # Loop over epochs
    loss_min = np.inf
    running_loss_valid = 0.0
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        optimizer, current_lr = exp_lr_scheduler(optimizer, epoch, init_lr=1e-3, lr_decay_epoch=10)
        running_loss_train = 0.0

        net.train()

        for i, data in enumerate(training_generator, 0):

            # get the inputs; data is a list of [inputs, labels]
            in_lr, in_2x, in_4x = data[0].to(device), data[1].to(device), data[2].to(device)

            # in_lr.requires_grad = True
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out_2x, out_4x = net(in_lr)
            loss_2x = criterion(out_2x, in_2x)
            loss_4x = criterion(out_4x, in_4x)

            loss = (loss_2x + loss_4x) / in_lr.shape[0]

            loss.backward()
            # loss_2x.backward(retain_graph=True)

            # loss_4x.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01 / current_lr)

            optimizer.step()

            # print statistics
            running_loss_train += loss.item()
            if i % 100 == 99:  # print every 5 mini-batches
                print('[%d, %5d] training loss: %.3f' %
                      (epoch + 1, i + 1, running_loss_train / 100))
                swanlab.log({'training_loss': running_loss_train / 100})
                running_loss_train = 0.0

        net.eval()

        for j, data_valid in enumerate(validation_generator, 0):
            in_lr, in_2x, in_4x = data_valid[0].to(device), data_valid[1].to(device), data_valid[2].to(device)

            out_2x, out_4x = net(in_lr)
            loss_2x = criterion(out_2x, in_2x)
            loss_4x = criterion(out_4x, in_4x)

            loss = (loss_2x + loss_4x) / in_lr.shape[0]

            running_loss_valid += loss.item()

        running_loss_valid = running_loss_valid / len(validation_generator)

        print('[%d] validation loss: %.3f' %
              (epoch + 1, running_loss_valid))
        swanlab.log({'validation_loss': running_loss_valid})

        if running_loss_valid < loss_min:
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': running_loss_valid,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_ckp(checkpoint, True, "ckp.pt", "best.pt")
            loss_min = running_loss_valid

        running_loss_valid = 0.0
    
    print('Finished Training')
