import torch
import numpy as np
import pandas as pd
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
from sklearn.preprocessing import MinMaxScaler

class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        '''
        inputs: dict([], [])
        outputs: [samples, output_features]
        '''
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return self.outputs.shape[0]

    def __getitem__(self, idx):
        entity = self.inputs['entity'][idx]
        sr = self.inputs['s_and_r'][idx]
        if 'entity_mask' in self.inputs.keys():
            entity_mask = self.inputs['entity_mask'][idx]
            return {'entity': entity, 's_and_r': sr, 'entity_mask': entity_mask, 'target': self.outputs[idx, :]}
        return {'entity': entity, 's_and_r': sr, 'target': self.outputs[idx, :]}


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def normalize_dataset(data, normalizer, column_wise=False, std=0):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            if std == 0:
                std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, state:str, normalizer = 'std', horizon=None):
    if horizon is None:
        horizon = args.horizon
    # NOTE: Load Entity (graph) data
    #load raw st dataset
    graph_data = load_st_dataset(args.graph_path)        # B, N, D
    #normalize st data
    graph_data, graph_scaler = normalize_dataset(graph_data, normalizer, args.column_wise)

    graph_data_train, graph_data_val, graph_data_test = split_data_by_ratio(graph_data, args.val_ratio, args.test_ratio)
    graph_data_train, graph_data_val, graph_data_test = split_data_by_ratio(graph_data, args.val_ratio, args.test_ratio)

    graph_x_train, graph_y_train = Add_Window_Horizon(graph_data_train, args.lag, horizon)
    graph_x_val, graph_y_val = Add_Window_Horizon(np.concatenate((graph_data_train[-args.lag:, ...], graph_data_val), axis=0), args.lag, horizon)
    graph_x_test, graph_y_test = Add_Window_Horizon(np.concatenate((graph_data_train[-args.lag:, ...], graph_data_test), axis=0), args.lag, horizon)

    # NOTE: Load Statistics and Regulation data
    if args.use_smooth_stats:
        state_df = pd.read_csv(f'{args.covid_dataset}/{state}_smooth.csv', index_col=[0])
        state_df['date'] = pd.to_datetime(state_df['date'])
        values = state_df[['smoothed_hosp', 'smoothed_cases', 'C7_Restrictions on internal movement_indicator_30_days_delay']].values
    else:
        state_df = pd.read_csv(f'{args.covid_dataset}/{args.state}.csv', index_col=[0])
        state_df = state_df.iloc[5:]
        state_df['date'] = pd.to_datetime(state_df['date'])
        values = state_df[['hospitalizations', 'cases', 'C7_Restrictions on internal movement_indicator_30_days_delay']].values

    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    sr_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = sr_scaler.fit_transform(values)

    sr_data_train, sr_data_val, sr_data_test = split_data_by_ratio(scaled, args.val_ratio, args.test_ratio)
    sr_x_train, sr_y_train = Add_Window_Horizon(sr_data_train, args.lag, horizon)
    sr_x_val, sr_y_val = Add_Window_Horizon(np.concatenate((sr_data_train[-args.lag:, ...], sr_data_val), axis=0), args.lag, horizon)
    sr_x_test, sr_y_test = Add_Window_Horizon(np.concatenate((sr_data_train[-args.lag:, ...], sr_data_test), axis=0), args.lag, horizon)

    # Make sure the length of each subset is the same
    print("TRAIN INPUT SHAPE:")
    print(len(graph_x_train), len(sr_x_train))
    print(graph_x_train.shape, sr_x_train.shape)
    print("TRAIN OUTPUT SHAPE:")
    print(len(graph_y_train), len(sr_y_train))
    print(graph_y_train.shape, sr_y_train.shape)
    assert len(graph_x_train) == len(sr_x_train)
    assert len(graph_y_train) == len(sr_y_train)
    assert len(graph_x_test) == len(sr_x_test)
    assert len(graph_y_test) == len(sr_y_test)
    assert len(graph_x_val) == len(sr_x_val)
    assert len(graph_y_val) == len(sr_y_val)

    # Combine the 2 type of data for dataloader
    x_train = {'entity': graph_x_train, 's_and_r': sr_x_train}
    x_val = {'entity': graph_x_val, 's_and_r': sr_x_val}
    x_test = {'entity': graph_x_test, 's_and_r': sr_x_test}

    y_train = sr_y_train
    y_val = sr_y_val
    y_test = sr_y_test
    ##############get dataloader######################
    train_dataset = TimeseriesDataset(x_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=False, drop_last=False)
    val_dataset = TimeseriesDataset(x_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, drop_last=False)
    test_dataset = TimeseriesDataset(x_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, graph_scaler, sr_scaler
