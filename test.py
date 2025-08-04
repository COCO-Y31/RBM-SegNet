import cv2
import os
import sys
import numpy as np
import datetime
import matplotlib as plt

from networks import *
from networks import Megnet
# from loss import SegmentationMetric
from dataset import DTM_Dataset
from preprocess import *
from data import *
import skimage.io as io

torch.backends.cudnn.benchmark = True  
torch.cuda.empty_cache 
interp = torch.nn.Upsample(size=(512, 512), mode='bilinear')

class Logger(object):
    def __init__(self, filename='./logs/all_info.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def dem2model_input(versions, dem_path, label_path=None, fine_size=512, padding=4):
    dem_map, proj, geotrans, data_type = readtif(dem_path)  
    dem_img = dem_map[padding:padding * (-1), padding:padding * (-1)]

    dem = get_constant2onezero(dem_img)
    slope = get_slope(dem_map, padding)
    hillshade = get_hillsahde(dem_path, padding)
    aspect = get_aspect(dem_path,padding)

    if versions == '1.0':
        input_layer = [dem, slope, aspect, hillshade]  # V1.0

    x_list = []
    for layer in input_layer:
        x_list.append(np.expand_dims(layer, axis=0))
    x_numpy = np.concatenate(x_list, axis=0)  # (C, H, W)
    print(x_numpy.shape)
    offset = 0
    if min(x_numpy.shape[0], x_numpy.shape[0]) > fine_size:
        offset = int((min(x_numpy.shape[0], x_numpy.shape[0]) - fine_size) * 0.5)
    print('original_size :', dem_map.shape, 'drop_padding :', dem_img.shape,
          'crop_range :', offset, offset + fine_size)
    x_numpy = x_numpy[:, offset:offset + fine_size, offset:offset + fine_size]
    print(x_numpy.shape)
    x_numpy = np.expand_dims(x_numpy, axis=0)  # (N, C, H, W)

    if label_path is not None:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label[padding:padding * (-1), padding:padding * (-1)]
        label = np.expand_dims(label, axis=0)  # (N, H, W) for SegmentationMetric
        label = label[:, offset:offset + fine_size, offset:offset + fine_size]
        return torch.Tensor(x_numpy.astype(np.float32)), torch.IntTensor(label.astype(np.uint8))
    else:
        return torch.Tensor(x_numpy.astype(np.float32)), None


def semantic_segmentation_test(save_dir, net_name, versions, use_half_training, model, device_id, test_data):
    """
    model_output : tensor (N, C, H, W)        N = 1, C = class_num
    """
    if use_half_training:
        versions = versions + '_half'
    NAME = net_name + '_' + versions
    model_param_path = os.path.join(save_dir, NAME + "_val_best_FWIoU.pth")
    model.load_state_dict(torch.load(model_param_path, map_location='cpu'))

    if device_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(device_id))
        print("Inference, cuda_is_available. use GPU:", device, NAME)
        model.to(device)
    else:
        device = torch.device("cpu")
        print("Inference, use CPU. model is:", NAME)
        model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model: {} Total_params: {}".format(net_name, pytorch_total_params))
    model.eval()

    torch.no_grad()
    print('Inference :', datetime.datetime.now())
    output = model(test_data)

    print('test_output',output)
    # output = interp(output)
    print(' ... \nFinished :', datetime.datetime.now())
    return output.detach().cpu()

def do_test(versions):
    in_channel, classNum = -1, -1
    if versions == '1.0':
        in_channel, classNum = 4, 2

    net_name = 'RBMSegNet'
    save_dir = r'./weights'
    use_half_training = False
    if net_name == 'Segnet':
        model = SegNet(input_channels=in_channel, output_channels=classNum)
    elif net_name == 'RBMSegNet':
        model = RBMSegNet(input_nbr=in_channel, label_nbr=classNum)
    else:
        model = UNet(input_channels=in_channel, output_channels=classNum)
    device_id = -1 # -1 means using CPU
    dem_path = r'./sample/DEM/test1.tif'

    label_path = r'./sample/label/test1.png'

    test_data, test_label = dem2model_input(versions, dem_path, label_path)

    model_output = semantic_segmentation_test(save_dir, net_name, versions, use_half_training, model, device_id,
                                              test_data)

    return model_output, test_data, test_label

def finalResult_show(versions, model_output, test_data, test_label):
    input_layer = np.array(torch.squeeze(test_data, dim=0))  # (C, H, W)

    model_output = torch.squeeze(model_output.cpu(), dim=0)  # (C, H, W)
    model_output = torch.nn.functional.softmax(model_output, dim=0)
    model_output = np.array(model_output)  # fine P_val numpy (C, H, W)

    pred_img = np.argmax(model_output, axis=0)  # (H, W)

    fig = plt.figure(figsize=(15, 10))

    if test_label is not None:
        label = np.array(torch.squeeze(test_label, dim=0))  # (H, W)
        plt.subplot(3, 3, 7)
        plt.imshow(label), plt.colorbar(shrink=0.8)
        plt.title("label")
    plt.subplot(3, 3, 8)
    plt.imshow(pred_img), plt.colorbar(shrink=0.8)
    plt.title("predict_result")  # ,plt.xticks([]), plt.yticks([])

    plt.savefig('sample_test1.png')
    plt.show()


if __name__ == '__main__':
    versions = '1.0'
    model_output, test_data, test_label = do_test(versions)
    finalResult_show(versions, model_output, test_data, test_label)

