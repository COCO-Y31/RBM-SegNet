import logging
import torch
import numpy as np
import datetime

from networks import *
from loss import dice_bce_loss, SegmentationMetric, structure_loss
from dataset import DTM_Dataset, get_data_list, get_data_lista, get_train_list,get_val_list,get_test_list
from data import *
from torch.utils.tensorboard import SummaryWriter
from early_stop import EarlyStopping

writer = SummaryWriter()

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache

log_handler = logging.getLogger('test_logger')
log_handler.setLevel(logging.INFO)
test_log = logging.FileHandler(r"./logs/{}.txt".format(str(datetime.datetime.now()).replace(':', '_')), 'a', encoding='utf-8')
test_log.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
test_log.setFormatter(formatter)
sh.setFormatter(formatter)
log_handler.addHandler(test_log)
log_handler.addHandler(sh)


def semantic_segmentation_train(net_name, versions, use_half_training, model, device_id, old_lr, resume, total_epoch,
                                train_data_loader, val_data_loader):
    if use_half_training:
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
        versions = versions + '_half'
    NAME = net_name + '_' + versions  # model_name for saving model weights
    device = None
    if device_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(device_id))
        log_handler.info("Training, cuda_is_available. use GPU:{}, {}".format(device, NAME))
    else:
        device = torch.device("cpu")
        log_handler.info("Training, use CPU. model is:".format(NAME))
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_handler.info("Model: {} Total_params: {}".format(net_name, pytorch_total_params))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=old_lr, eps=1e-8)  
    loss_func = dice_bce_loss()

    if resume > 1:
        log_handler.info("Resume training from epoch {}".format(resume))
        model.load_state_dict(torch.load('./weights/' + NAME + '_lastest_Lr.pth'))
    try:
        all_train_loss, all_val_loss, all_train_FWIoU, all_val_FWIoU = [], [], [], []
        epoch = 1
        no_optim, no_val_optim, train_epoch_best_loss, val_epoch_best_loss, train_epoch_best_FWIoU, val_epoch_best_FWIoU = 0, 0, 9999, 9999, 0, 0
        for epoch in range(0 + resume, total_epoch + 1):
            dt_size = len(train_data_loader.dataset)
            Iterations = (dt_size - 1) // train_data_loader.batch_size + 1  # drop_last=True, do not add 1
            print_frequency = Iterations // 10 
            gap_frequency = 500000
            log_handler.info(
                'Train_Epoch = [%3d/%3d] | Iterations = %3d | LearningRate = %.8f | %s' % (epoch, total_epoch,
                                                                                           Iterations,
                                                                                           optimizer.state_dict()[
                                                                                               'param_groups'][0]['lr'],
                                                                                           str(datetime.datetime.now())))
            model.train()
            train_epoch_loss, train_epoch_FWIoU, train_evalus_res = 0, 0, None
            print_frequency_count = 0
            for i, (img, mask) in enumerate(train_data_loader):
                b_x = img.to(device)
                b_y = mask.to(device)
                if not use_half_training:
                    optimizer.zero_grad()
                    output = model(b_x)
                    loss = loss_func(output, b_y, use_half_training)
                    loss.backward()
                    optimizer.step()

                if use_half_training:
                    optimizer.zero_grad()
                    with autocast():
                        output = model(b_x)
                        loss = loss_func(output, b_y, use_half_training)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if i % print_frequency == 0 and i > 0 and i // gap_frequency < 10:
                    log_handler.info('\ttrain_epoch = %3d | iters = %3d | loss = %3.8f | %s' % (
                    epoch, i, loss.item(), str(datetime.datetime.now())))
                    metric = SegmentationMetric(numClass=output.shape[1], ignore_labels=None)
                    train_evalus_res = metric.evalus(torch.argmax(output, dim=1).cpu(), torch.argmax(b_y, dim=1).cpu())
                    log_handler.info(train_evalus_res)
                    train_iters_FWIoU = float(train_evalus_res.split(':')[-1])
                    all_train_loss.append(np.around(loss.item(), 5))
                    all_train_FWIoU.append(np.around(train_iters_FWIoU, 5))
                    print_frequency_count += 1
                    train_epoch_FWIoU += train_iters_FWIoU
                train_epoch_loss += loss.item() * b_x.size(0)
            train_epoch_loss /= dt_size
            train_epoch_FWIoU /= print_frequency_count
            log_handler.info("{}_Train_Epoch_{}_AvgLoss_{}_FWIoU_{}".format(str(datetime.datetime.now()), epoch,
                                                                            np.around(train_epoch_loss, 5),
                                                                            np.around(train_epoch_FWIoU, 5)))
            # -------------------------------------------------------------------------------------------------------
            # --------------------------------  model.eval()  -------------------------------------------------------
            # -------------------------------------------------------------------------------------------------------
            dt_size = len(val_data_loader.dataset)
            Iterations = (dt_size - 1) // val_data_loader.batch_size + 1  # drop_last=True, do not add 1
            log_handler.info('Val_Epoch = [%3d/%3d] | Iterations = %3d | LearningRate = %.8f | %s' % (
            epoch, total_epoch, Iterations, optimizer.state_dict()['param_groups'][0]['lr'],
            str(datetime.datetime.now())))
            model.eval()  
            torch.no_grad()  

            val_epoch_loss, val_epoch_FWIoU, val_evalus_res = 0, 0, None
            print_frequency_count = 0
            for i, (img, mask) in enumerate(val_data_loader):
                b_x = img.to(device)
                b_y = mask.to(device)
                if not use_half_training:
                    output = model(b_x)
                    loss = loss_func(output, b_y, use_half_training)

                if i % print_frequency == 0 and i > 0 and i // gap_frequency < 10:
                    log_handler.info('\tval_epoch = %3d | iters = %3d | loss = %3.8f | %s' % (
                    epoch, i, loss.item(), str(datetime.datetime.now())))
                    metric = SegmentationMetric(numClass=output.shape[1], ignore_labels=None)
                    val_evalus_res = metric.evalus(torch.argmax(output, dim=1).cpu(), torch.argmax(b_y, dim=1).cpu())
                    log_handler.info(val_evalus_res)
                    val_iters_FWIoU = float(val_evalus_res.split(':')[-1])
                    all_val_loss.append(np.around(loss.item(), 5))
                    all_val_FWIoU.append(np.around(val_iters_FWIoU, 5))
                    print_frequency_count += 1
                    val_epoch_FWIoU += val_iters_FWIoU
                val_epoch_loss += loss.item() * b_x.size(0)
            val_epoch_loss /= dt_size
            val_epoch_FWIoU /= print_frequency_count
            log_handler.info("{}_Val_Epoch_{}_AvgLoss_{}_FWIoU_{}".format(str(datetime.datetime.now()), epoch,
                                                                          np.around(val_epoch_loss, 5),
                                                                          np.around(val_epoch_FWIoU, 5)))
            # torch.save(model.state_dict(), "./weights/{}_epoch_{}.pth".format(NAME, epoch)) 
            # -------------------------------------------------------------------------------------------------------
            # --------------------------------  early stop training trackers  ---------------------------------------
            # ------------ (1) training loss not decrease[0.01] in 3 epochs -> update LR and loop (1) in 6 times
            # ------------ (2) val FWIoU not increase[0.01] in 3 epochs -> directly stop
            # -------------------------------------------------------------------------------------------------------
            if train_epoch_loss >= train_epoch_best_loss + 0.00:
                no_optim += 1
            else:
                no_optim = 0
                train_epoch_best_loss = train_epoch_loss
                torch.save(model.state_dict(), './weights/' + NAME + '_lastest_Lr.pth')
            if no_optim > 6:
                log_handler.info('The learningRate has been optimised 6 times. Training_Early_Stop ...')
                break
            if no_optim >= 3:
                if old_lr < 5e-7:
                    log_handler.info(
                        "{}_EPOCH_{}_smallest_LR_Early_Stop ...".format(str(datetime.datetime.now()), epoch))
                    break
                model.load_state_dict(torch.load('./weights/' + NAME + '_lastest_Lr.pth'))
                new_lr = old_lr / 5 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                log_handler.info(
                    "{}_Update_Learning_Late_{}_To_{}".format(str(datetime.datetime.now()), old_lr, new_lr))
                old_lr = new_lr
            # ---------------------------------------------------------------------------
            if val_epoch_FWIoU <= val_epoch_best_FWIoU - 0.00:
                no_val_optim += 1
                if no_val_optim >= 7:
                    log_handler.info(
                        "{}_EPOCH_{}_best_FWIoU_Early_Stop ...".format(str(datetime.datetime.now()), epoch))
                    break
            else:
                no_val_optim = 0
                val_epoch_best_FWIoU = val_epoch_FWIoU
                torch.save(model.state_dict(), './weights/' + NAME + '_val_best_FWIoU.pth')
            # end of one epoch
        # end of all epoch
    finally:  # final save checkpoint
        log_handler.info(
            "{}_EPOCH_{}_Porgress_Interrupt or Training_Early_Stop ...".format(str(datetime.datetime.now()), epoch))
        # torch.save(model.state_dict(), "./weights/{}_Interrupt_epoch_{}.pth".format(NAME, epoch))
        log_handler.info(
            "\nall_train_loss = {}\n, all_val_loss = {}\n, all_train_FWIoU = {}\n, all_val_FWIoU = {}\n".format(
                all_train_loss, all_val_loss, all_train_FWIoU, all_val_FWIoU))

if __name__ == '__main__':
    in_channel, classNum = 4, 2
    net_name = 'RBMSegNet'
    versions = '1.0'
    use_half_training = False

    device_id = -1
    old_lr = 1e-4  # init learning rate
    resume = 1  # resume > 1: torch.load('./weights/'+ NAME + '_lastest_Lr.pth'))
    total_epoch = 100
    base_batch_size = 8

    if net_name == 'Segnet':
        model = SegNet(input_channels=in_channel, output_channels=classNum)
    elif net_name == 'RBMSegNet':
        model = RBMSegNet(input_nbr=in_channel, label_nbr=classNum)
    else:
        model = UNet(input_channels=in_channel, output_channels=classNum)
    device_id = -1  # -1 means using CPU

    log_handler.info("in_channel : {}, classNum : {}, net_name : {}, versionsm : {}, use_half_training : {}, device_id : {}, old_lr : {}, resume : {}, total_epoch : {}".format(
            in_channel, classNum, net_name, versions, use_half_training, device_id, old_lr, resume, total_epoch))

    dem_data_folder = r''
    target_data_folder = r''

    all_data_list = get_data_list(dem_data_folder, target_data_folder)
    from sklearn.model_selection import train_test_split
    train_data_list, val_data_list = train_test_split(all_data_list, train_size=0.8, test_size=0.2)
    train_dataSet = DTM_Dataset(train_data_list, fine_size=[512, 512], num_classes=classNum)
    val_dataSet = DTM_Dataset(val_data_list, fine_size=[512, 512], num_classes=classNum)
    train_data_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=base_batch_size * 4, shuffle=True,
                                                    drop_last=True,
                                                    num_workers=0)

    val_data_loader = torch.utils.data.DataLoader(val_dataSet, batch_size=base_batch_size, shuffle=True, drop_last=True,
                                                  num_workers=0)
    log_handler.info("train_data_loader dataset {}".format(len(train_data_list)))
    log_handler.info("val_data_loader dataset {}".format(len(val_data_list)))
    
    semantic_segmentation_train(net_name, versions, use_half_training, model, device_id, old_lr, resume, total_epoch,
                                train_data_loader, val_data_loader)
