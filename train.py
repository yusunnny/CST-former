import os
import sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plot
from cls import cls_feature_class as cls_feature_class, cls_data_generator as cls_data_generator
import parameters as parameters
import time
import json
from time import gmtime, strftime
import torch
import torch.optim as optim
from utility.graphs import draw_loss

plot.switch_backend('agg')
from cls.cls_compute_seld_results import ComputeSELDResults
from utility.load_state_dict import load_state_dict
from architecture import CST_former_model as model_architecture
from utility.test_epoch import test_epoch
from utility.train_epoch import train_epoch as train_epoch
from utility.loss_adpit import MSELoss_ADPIT

def main(argv):
    """
    Main wrapper for training sound event localization and detection network.
    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1
    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inpluts for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    # os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    device = torch.device('cuda')

    #---------------------------------------------- (For Reproducibility)
    # fix the seed for reproducibility
    seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    # ----------------------------------------------
    torch.autograd.set_detect_anomaly(True)

    # use parameter set defined by user
    task_id = '33' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        if '2020' in params['dataset_dir']:
            test_splits = [1]
            val_splits = [2]
            train_splits = [[3, 4, 5, 6]]

        elif '2021' in params['dataset_dir']:
            test_splits = [6]
            val_splits = [5]
            train_splits = [[1, 2, 3, 4]]

        elif '2022' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]]
        elif '2023' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1,2,3]]
        else:
            print('ERROR: Unknown dataset splits')
            exit()

    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print(
            '------------------------------------      SPLIT {}   -----------------------------------------------'.format(
                split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        loc_feat = params['dataset']
        if params['dataset'] == 'mic':
            if params['use_salsalite']:
                loc_feat = '{}_salsa'.format(params['dataset'])
            else:
                loc_feat = '{}_gcc'.format(params['dataset'])
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        # ----------------------------------------------
        unique_name = '{}_{}_{}_split{}_{}_{}'.format(
            task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat
        )
        cls_feature_class.create_folder(os.path.join(params['save_dir'], unique_name, params['model_dir']))
        model_name = os.path.join(params['save_dir'], unique_name, params['model_dir'], 'model.h5')
        print("unique_name: {}\n".format(unique_name))
        # ----------------------------------------------

        # Load train and validation data
        print('Loading training dataset:')
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True
        )

        # Collect i/o data size and load model configuration
        data_in, data_out = data_gen_train.get_data_sizes()
        model = model_architecture.CST_former(data_in, data_out, params)

        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        if params['finetune_mode']:
            print('Running in finetuning mode.')
            model = load_state_dict(model, params['pretrained_model_weights'])
        print('---------------- SELD-net -------------------')
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
        print(
            'MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n, rnn_size: {}\n, nb_attention_blocks: {}\n, fnn_size: {}\n'.format(
                params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'],
                params['rnn_size'], params['nb_self_attn_layers'],
                params['fnn_size']))
        print(model)

        # Dump results in DCASE output format for calculating final scores
        # ----------------------------------------------
        dcase_output_folder = os.path.join(params["save_dir"], unique_name,params['dcase_output_dir'], strftime("%Y%m%d%H%M%S", gmtime()))
        # ----------------------------------------------
        dcase_output_val_folder = os.path.join(dcase_output_folder, 'val')
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        print('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder))

        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)

        # start training
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999
        patience_cnt = 0

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        if params['lr'] is None:  # only base_lr is specified
            params['lr'] = params['blr'] * params['batch_size'] / 256

        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        # loss preparation
        if params['multi_accdoa'] is True:
            criterion = MSELoss_ADPIT()
        else:
            criterion = nn.MSELoss()

        # Start Train_Valid Loss recording
        train_loss_rec = np.empty([params["nb_epochs"]])
        valid_loss_rec = np.empty([params["nb_epochs"]])
        valid_seld_scr_rec = np.empty([params["nb_epochs"]])
        valid_ER_rec = np.empty([params["nb_epochs"]])
        valid_F_rec = np.empty([params["nb_epochs"]])
        valid_LE_rec = np.empty([params["nb_epochs"]])
        valid_LR_rec = np.empty([params["nb_epochs"]])
        learning_rate_rec = np.empty([params["nb_epochs"]])

        for epoch_cnt in range(nb_epoch):
            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            train_loss, learning_rate, = train_epoch(data_gen_train, optimizer, model, criterion,
                                                         params, device, epoch_cnt)

            train_time = time.time() - start_time

            # ---------------------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------------------
            start_time = time.time()
            val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
            # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
            val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(
                dcase_output_val_folder)

            val_time = time.time() - start_time

            # Save model if loss is good
            if val_seld_scr <= best_seld_scr:
                best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr
                torch.save(model.state_dict(), model_name)

            # Print stats
            print(
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                'lr:{:0.4f},'
                'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                'ER/F/LE/LR/SELD: {}, '
                'best_val_epoch: {} {}'.format(
                    epoch_cnt, train_time, val_time,
                    learning_rate,
                    train_loss, val_loss,
                    '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr),
                    best_val_epoch,
                    '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE, best_LR,
                                                                       best_seld_scr))
            )

            log_stats = {'epoch': epoch_cnt,
                         'lr':learning_rate,
                        'train_loss': train_loss,
                         'valid_loss': val_loss,
                         'val_ER': val_ER, 'val_F': val_F, 'val_LE': val_LE, 'val_LR': val_LR,
                         'val_seld_scr': val_seld_scr,}

            with open(os.path.join(dcase_output_folder, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            train_loss_rec, valid_loss_rec, valid_seld_scr_rec, valid_ER_rec, valid_F_rec, valid_LE_rec, valid_LR_rec, learning_rate_rec \
                = draw_loss(dcase_output_folder, epoch_cnt, best_val_epoch, learning_rate,
                                 train_loss, val_loss, val_seld_scr,val_ER, val_F, val_LE, val_LR,
                                 train_loss_rec, valid_loss_rec, valid_seld_scr_rec,
                                 valid_ER_rec, valid_F_rec, valid_LE_rec, valid_LR_rec, learning_rate_rec)
            patience_cnt += 1
            if patience_cnt > params['patience']:
                break

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        print('Load best model weights')
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        print('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
        )

        # Dump results in DCASE output format for calculating final scores
        dcase_output_test_folder = os.path.join(dcase_output_folder, 'test')
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

        _ = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)
        use_jackknife = True
        test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(
            dcase_output_test_folder, is_jackknife=use_jackknife)
        print('\nTest Loss')
        print('SELD score (early stopping metric): {:0.2f} {}'.format(
            test_seld_scr[0] if use_jackknife else test_seld_scr,
            '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
        print(
            'SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(test_ER[0] if use_jackknife else test_ER,
                                                                              '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0],
                                                                                                          test_ER[1][
                                                                                                              1]) if use_jackknife else '',
                                                                              100 * test_F[
                                                                                  0] if use_jackknife else 100 * test_F,
                                                                              '[{:0.2f}, {:0.2f}]'.format(
                                                                                  100 * test_F[1][0], 100 * test_F[1][
                                                                                      1]) if use_jackknife else ''))
        print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(
            test_LE[0] if use_jackknife else test_LE,
            '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '',
            100 * test_LR[0] if use_jackknife else 100 * test_LR,
            '[{:0.2f}, {:0.2f}]'.format(100 * test_LR[1][0], 100 * test_LR[1][1]) if use_jackknife else ''))
        if params['average'] == 'macro':
            print('Classwise results on unseen test data')
            print('Class\tER\tF\tLE\tLR\tSELD_score')
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                    cls_cnt,
                    classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                                                classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
