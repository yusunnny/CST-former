# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.

def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=False,
        # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='output/2022/261_4_dev_split0_accdoa_foa/models/model.h5',

        # INPUT PATHnum
        dataset_dir='./data/2023DCASE_data/',  # Base folder containing the foa/mic and metadata folders

        # OUTPUT PATHS
        feat_label_dir='./data/feature_labels_2023/', # Directory to dump extracted features and labels

        save_dir = 'output/2023', # 'output/2022', 'output/2023'
        model_dir='models/',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results/',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,
        audio_overlap = False,

        use_real_imag = False,
        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,
        ACS = False,
        FoA16Rotation = False,

        # MODEL TYPE
        baseline = True,
        encoder = 'conv',           # ['conv', 'ResNet', 'SENet']
        LinearLayer = False,        # Linear Layer right after attention layers (usually not used/employed in baseline model)
        FreqAtten = False,          # Use of Divided Spectro-Temporal Attention (DST Attention)
        ChAtten_DCA = False,        # Use of Divided Channel-S-T Attention (CST Attention)
        ChAtten_ULE = False,        # Use of Divided C-S-T attention with Unfold (Unfolded CST attention)
        CMT_block = False,          # Use of LPU & IRFNN
        CMT_split = False,          # Apply LPU & IRFNN on S, T attention layers independently
        multi_accdoa=False,         # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,            # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.


        # DNN MODEL PARAMETERS
        label_sequence_length=50,  # Feature sequence length
        batch_size=128,  # Batch size0
        dropout_rate=0.05,  # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,  # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],
        t_pooling_loc = 'front',
        # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=500,  # Train for maximum epochs

        # Learning Rate Scheduler
        lr_scheduler = False,
        lr_by_epoch = False,
        lr_ramp = False,
        lr=1e-3,
        min_lr=1e-6,
        blr=1e-3,
        warmup_epochs=5,

        # METRIC
        average='macro',  # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,
    )


    params['feature_label_resolution'] = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * params['feature_label_resolution']
    params['t_pool_size'] = [params['feature_label_resolution'], 1, 1]  # CNN time pooling
    # params['t_pool_size'] = [1, 1, feature_label_resolution]
    params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")
        
    elif argv == '21':
        print("[DST-attention] FOA + ACCDOA + Divided S-T (S dim : 16)\n")
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

        params['baseline'] = False
        params['FreqAtten'] = True
        params["f_pool_size"] = [2, 2, 1]

    elif argv == '31':
        print("[DST-former] FOA + Multi-ACCDOA + DST + CMT (S dim : 16)\n")
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

        # params['dataset_dir'] = './data/2022DCASE_data/'
        # params['feat_label_dir'] = './data/feature_labels_2022/'
        # params['save_dir'] = 'output/2022'

        params['baseline'] = False
        params['FreqAtten'] = True
        params['CMT_block'] = True  # Use of LPU & IRFNN

        params["f_pool_size"] = [1,2,2]
        params['t_pool_size'] = [1,1, params['feature_label_resolution']]

        params['batch_size'] = 256 #256

    elif argv == '32':
        print("[CST-former: Divided Channel Attention] FOA + Multi-ACCDOA + CST_DCA + CMT (S dim : 16)\n")
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

        params['lr_scheduler'] = True
        params['lr_by_epoch'] = True
        params['lr_by_epoch_stay_epoch'] = 150

        params['baseline'] = False
        params['FreqAtten'] = True
        params['ChAtten_DCA'] = True
        params['CMT_block'] = True

        params["f_pool_size"] = [2, 2, 1]
        params['t_pool_size'] = [params['feature_label_resolution'], 1, 1]
        params['batch_size'] = 32

    elif argv == '33':
        print("[CST-former: Unfolded Local Embedding] FOA + Multi-ACCDOA + CST Unfold + CMT (S dim : 16)\n")
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

        params['baseline'] = False
        params['lr_scheduler'] = True
        params['lr_by_epoch'] = True
        params['lr_by_epoch_stay_epoch'] = 150 #150
        params['nb_epochs'] = 300
        params['batch_size'] = 256 #256

        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True

        params["f_pool_size"] = [1,2,2] 
        params['t_pool_size'] = [1,1, params['feature_label_resolution']]

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
        params['average'] = 'micro'
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13
    else:
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
