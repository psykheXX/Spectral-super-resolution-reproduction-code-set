import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os
from thop import profile, clever_format
import time

from load_dataset import TrainDataset, ValidDataset
from HSCNND import HSCNND
from utils import Loss_MRAE, Loss_PSNR, Loss_RMSE, AverageMeter

def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

if __name__ == '__main__':
    batch_size = 16
    per_epoch_iteration = 1000
    epoch_num = 300
    total_iteration = per_epoch_iteration * epoch_num
    learning_rate = 4e-4
    weight_decay = 1e-4
    momentum = 0.9
    early_stopping_patience = 30

    train_dataset = TrainDataset(patch_size=128)
    valid_dataset = ValidDataset()

    model = HSCNND(in_channels=3, FMN_num=38)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m == model.conv1R:
                init.normal_(m.weight, mean=0.0, std=0.001)
            else:
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                # Initialize biases to zero
                init.constant_(m.bias, 0)

    criterion_mrae = Loss_MRAE()
    criterion_psnr = Loss_PSNR()
    criterion_rmse = Loss_RMSE()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.to(device)
        criterion_mrae.to(device)
        criterion_rmse.to(device)
        criterion_psnr.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(momentum, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

    cudnn.benchmark = True
    iteration = 0
    best_val_loss = float('inf')
    iteration_no_improvement = 0
    early_stopping_occur = None

    # TensorBoard summary writer
    log_dir = "/root/tf-logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    Loss_list = []
    train_stop = None
    while iteration < total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2,
                                pin_memory=True, drop_last=True)
        for i, (image, label) in enumerate(train_loader):
            iteration_start_time = time.time()
            label = label.to(device)
            image = image.to(device)
            image = Variable(image)
            label = Variable(label)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(image)
            loss = criterion_mrae(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration + 1
            iteration_time = time.time() - iteration_start_time
            print(f'Iteration {iteration} time: {iteration_time:.2f} seconds')

            writer.add_scalar('Training Loss', losses.avg, iteration)
            writer.add_scalar('Learning Rate', lr, iteration)
            
            if iteration % batch_size == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                      % (iteration, total_iteration, learning_rate, losses.avg))
                Loss_list.append(losses.avg)
            
            # if losses.avg < best_val_loss:
            #     best_val_loss = losses.avg
            #     iteration_no_improvement = 0
            # else:
            #     iteration_no_improvement += 1

            # if iteration_no_improvement >= early_stopping_patience:
            #     early_stopping_occur = True
            #     print(f'Early stopping at iteration {iteration}')
            #     torch.save(model.state_dict(), '/root/code/best_model.pth')
            #     mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
            #     print(f'Validation: MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
            #     break

            if iteration % total_iteration == 0:
                torch.save(model.state_dict(), '/root/code/best_model.pth')
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                train_stop = True
                break
        if train_stop
            break

        # if early_stopping_occur:
        #     break

    writer.close()
    
    input_sample = torch.randn(1, 3, 482, 512).to(device)
    macs, params = profile(model, inputs=(input_sample,), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    print(f'Params (M): {params}')
    print(f'FLOPS (G): {macs}')

    # Save train_loss curve as list in a new file
    with open('/root/code/train_loss.txt', 'w') as f:
        for loss in Loss_list:
            f.write(f"{loss}\n")











