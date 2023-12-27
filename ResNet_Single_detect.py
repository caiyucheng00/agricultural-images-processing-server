import os
from functions import *
import time


def detect_forward_phe(data_path, save_model_path, res_size, k):
    data_path = data_path
    save_model_path = save_model_path
    res_size = res_size
    k = k
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768

    # 图片转化tensor
    fnames = sorted(os.listdir(data_path))
    all_X_list = []
    for f in fnames:
        all_X_list.append(f)

    # data loading parameters
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # reset data loader
    all_data_params = {'shuffle': False, 'num_workers': 4, 'pin_memory': True}
    all_data_loader = data.DataLoader(Dataset_SingleCNN_Detect(data_path, all_X_list, transform=transform),
                                      **all_data_params)

    # Create model
    resnet = ResNet50Encoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, num_classes=k,
                             pretrained=False).to(device)
    resnet.load_state_dict(torch.load(os.path.join(save_model_path, 'phe.pth')))
    print('resnet model reloaded!')

    # make all video predictions by reloaded model
    print('Predicting all {} images:'.format(len(all_data_loader.dataset)))
    all_y_pred = Single_final_prediction(resnet, device, all_data_loader)  # list[0,1]
    return all_y_pred


def detect_forward_scene(data_path, save_model_path, res_size, k):
    data_path = data_path
    save_model_path = save_model_path
    res_size = res_size
    k = k
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768

    # 图片转化tensor
    fnames = sorted(os.listdir(data_path))
    all_X_list = []
    for f in fnames:
        all_X_list.append(f)

    # data loading parameters
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # reset data loader
    all_data_params = {'shuffle': False, 'num_workers': 4, 'pin_memory': True}
    all_data_loader = data.DataLoader(Dataset_SingleCNN_Detect(data_path, all_X_list, transform=transform),
                                      **all_data_params)

    # Create model
    alexnet = AlexNetEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, num_classes=k,
                             pretrained=False).to(device)
    alexnet = nn.DataParallel(alexnet)
    check_point = torch.load(os.path.join(save_model_path, 'scene.pth'))
    alexnet.load_state_dict(check_point)
    print('alexnet model reloaded!')

    # make all video predictions by reloaded model
    print('Predicting all {} images:'.format(len(all_data_loader.dataset)))
    all_y_pred = Single_final_prediction(alexnet, device, all_data_loader)  # list[0,1]
    return all_y_pred


if __name__ == '__main__':
    start_time = time.time()
    data_path = '/data2/caiyucheng/project/phenology-detection/show_single/'
    save_model_path = 'exp/exp04/'
    res_size = 224
    k = 9

    detect_forward_phe(data_path, save_model_path, res_size, k)
    end_time = time.time()
    exe_time = end_time - start_time
    print("Time:", exe_time)
