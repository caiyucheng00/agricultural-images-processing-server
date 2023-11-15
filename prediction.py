from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle

def detect_forword():

    # set path
    data_path = "./show_time/"                # define UCF-101 RGB data path
    action_name_path = "./winter_wheat.pkl"
    save_model_path = "./phe_models/"

    # use same encoder CNN saved!
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512   # latent dim extracted by 2D CNN
    res_size = 500        # ResNet image size
    dropout_p = 0.3       # dropout probability

    # use same decoder RNN saved!
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 512
    RNN_FC_dim = 256

    # training parameters
    k = 9             # number of target category
    batch_size = 210
    # Select which frame to begin & end in videos
    begin_frame, end_frame, skip_frame = 1, 31, 1


    with open(action_name_path, 'rb') as f:
        action_names = pickle.load(f)   # load UCF101 actions names

    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)

    # show how many classes there are
    print(list(le.classes_))

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    actions = []
    fnames = os.listdir(data_path)

    all_names = []
    for f in fnames:
        if f.find('3_') == 0:
            actions.append('1Three-leaf')
        elif f.find('4_') == 0:
            actions.append('2Four-leaf')
        elif f.find('5_') == 0:
            actions.append('3Five-leaf')
        elif f.find('6_') == 0:
            actions.append('4Jointing')
        elif f.find('7_') == 0:
            actions.append('5Booting')
        elif f.find('8_') == 0:
            actions.append('6Heading')
        elif f.find('9_') == 0:
            actions.append('7Anthesis')
        elif f.find('a_') == 0:
            actions.append('8Filling')
        elif f.find('b_') == 0:
            actions.append('9Maturity')

        all_names.append(f)

    # list all data files
    all_X_list = all_names              # all video file names
    all_y_list = labels2cat(le, actions)    # all video labels
    all_y_list = [0]

    # data loading parameters
    use_cuda = False
    device = torch.device("cpu")   # use CPU or GPU
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True, 'drop_last': True} if use_cuda else {}

    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    # reset data loader
    all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    all_data_loader = data.DataLoader(Dataset_CRNN(data_path, all_X_list, all_y_list, selected_frames, transform=transform), **all_data_params)


    # reload CRNN model
    cnn_encoder = ResNet50Encoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)
    cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch99.pth'), map_location='cpu'))
    rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch99.pth'), map_location='cpu'))
    print('CRNN model reloaded!')


    # make all video predictions by reloaded model
    print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
    all_y_pred = CRNN_final_prediction([cnn_encoder, rnn_decoder], device, all_data_loader)
    index = int(all_y_pred)
    labels = ['三叶期', '四叶期', '五叶期', '拔节期', '孕穗期', '抽穗期', '开花期', '灌浆期', '成熟期']
    phenology_name = labels[index]
    print(phenology_name)
    
    return phenology_name
