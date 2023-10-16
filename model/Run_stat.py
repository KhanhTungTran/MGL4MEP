import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import numpy as np
import argparse
import configparser
from lib.TrainInits import init_seed
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from lib.metrics import All_Metrics
from lib.dataloader import split_data_by_ratio
from statsmodels.tsa.arima.model import ARIMA


def moving_average(a, n=7):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#*************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'COVID'      #PEMSD4 or PEMSD8
COVID_DATASET = '../data/'
DEVICE = 'cuda:0'

#get configuration
config_file = './{}_{}.conf'.format(DATASET)
print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

from lib.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--covid_dataset', default=COVID_DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--cuda', default=True, type=bool)
#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
#model
args.add_argument('--model_type', default=config['model']['model_type'], type=str)
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--input_shape', default=config['model']['input_shape'], type=int)
args.add_argument('--sr_input_dim', default=config['model']['sr_input_dim'], type=int)
args.add_argument('--s_input_dim', default=config['model']['s_input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
args.add_argument('--use_mask', default=config['model']['use_mask'], type=eval)
#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--use_best', default=config['test']['use_best'], type=bool)
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--wandb_project', default=config['log']['wandb_project'], type=str)
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)

args.add_argument('--model_id', type=str)
args = args.parse_args()
init_seed(args.seed)


# NOTE: Load Statistics and Regulation data
state_df = pd.read_csv(f'{args.covid_dataset}/New York.csv', index_col=[0])
# state_df = pd.read_csv(f'{args.covid_dataset}/california.csv', index_col=[0])
state_df = state_df.iloc[5:]
state_df['date'] = pd.to_datetime(state_df['date'])
values = state_df[['hospitalizations', 'cases']].values
# ensure all data is float
values = values.astype('float32')
# normalize features
mean = values.mean()
std = values.mean()
sr_scaler = MinMaxScaler(feature_range=(0, 1))
scaled = sr_scaler.fit_transform(values)

sr_data_train, sr_data_val, sr_data_test = split_data_by_ratio(scaled, args.val_ratio, args.test_ratio)
SHIFT = args.horizon - 1
if (args.model_type=="AVG"):
    avg = np.concatenate((sr_data_train, sr_data_val), axis=0).mean(axis=0)
    targets_lab = sr_scaler.inverse_transform(sr_data_test)
    print(targets_lab.shape, SHIFT)
    targets_lab = targets_lab[SHIFT:, :]
    print(targets_lab.shape)
    avg = np.repeat(np.expand_dims(avg, axis=0), targets_lab.shape[0], axis=0)
    y_pred = sr_scaler.inverse_transform(avg)

elif (args.model_type=="LAST_DAY"):
    preds = np.concatenate((np.expand_dims(sr_data_val[-1], axis=0), sr_data_test[:-(SHIFT+1)]), axis=0)
    targets_lab = sr_scaler.inverse_transform(sr_data_test)
    targets_lab = targets_lab[SHIFT:, :]
    print(targets_lab.shape)
    y_pred = sr_scaler.inverse_transform(preds)
    print(targets_lab[:5])
    print(y_pred[:5])

elif (args.model_type=="AVG_WINDOW"):
    preds = moving_average(np.concatenate((sr_data_val[-args.lag:, ...], sr_data_test[:-(SHIFT+1)]), axis=0), n=args.lag)
    targets_lab = sr_scaler.inverse_transform(sr_data_test)
    targets_lab = targets_lab[SHIFT:, :]
    print(preds.shape)
    print(targets_lab.shape)
    y_pred = sr_scaler.inverse_transform(preds)

elif (args.model_type=="ARIMA"):
    print(sr_data_train.shape)
    preds = []
    for j in range(sr_data_train.shape[1]):
        try:
            fit = ARIMA(sr_data_train[:,j], order=(2, 0, 2)).fit()
        except:
            fit = ARIMA(sr_data_train[:,j], order=(1, 0, 0)).fit()
        pred = abs(fit.predict(start=len(sr_data_train)+len(sr_data_val), end=len(sr_data_train)+len(sr_data_val)+len(sr_data_test)-(SHIFT+1)))
        preds.append(pred)
    preds = np.array(preds)
    preds = np.transpose(preds, (1, 0))
    print(preds.shape)
    targets_lab = sr_scaler.inverse_transform(sr_data_test)
    targets_lab = targets_lab[SHIFT: , :]
    print(targets_lab.shape)
    y_pred = sr_scaler.inverse_transform(preds)

mae, rmse, mape, _ = All_Metrics(y_pred, targets_lab, args.mae_thresh, args.mape_thresh)
print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            mae, rmse, mape*100))
