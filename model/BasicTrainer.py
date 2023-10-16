import torch
import math
import os
import time
import copy
import numpy as np
import wandb
from lib.logger import get_logger
from lib.metrics import All_Metrics

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model_type, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def update_loader(self, train_loader, val_loader, test_loader, scaler, horizon):
        self.curr_horizon = horizon
        self.scaler = scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(val_dataloader):
            for batch_idx, dct in enumerate(val_dataloader):
                dct['entity'] = dct['entity'].to(self.args.device).type(torch.float32)
                dct['s_and_r'] = dct['s_and_r'].to(self.args.device)
                dct['target'] = dct['target'].to(self.args.device)
                # data = data[..., :self.args.input_dim]
                label = dct['target'][..., :self.args.output_dim].squeeze()
                output = self.model(dct)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output.to(self.args.device), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        # val_loss = total_val_loss / len(val_dataloader)
        val_loss = total_val_loss
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, dct in enumerate(self.train_loader):
            dct['entity'] = dct['entity'].to(self.args.device).type(torch.float32)
            dct['s_and_r'] = dct['s_and_r'].to(self.args.device)
            dct['target'] = dct['target'].to(self.args.device)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.

            label = dct['target'][..., :self.args.output_dim].squeeze()
            output = self.model(dct)
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            loss = self.loss(output.to(self.args.device), label)
            loss.backward() 

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        # train_epoch_loss = total_loss/self.train_per_epoch
        train_epoch_loss = total_loss
        self.logger.info('**********Train Epoch {}: average Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)
            if self.val_loader == None or len(self.val_loader) == 0:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            wandb.log({
                f'train_loss': train_epoch_loss,
                f'valid_loss': val_epoch_loss,
                f'best_valid_loss': best_loss,
                f'epoch': epoch,
            })
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        if self.args.use_best:
            self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        x_inp = []
        with torch.no_grad():
            for batch_idx, dct in enumerate(data_loader):
                dct['entity'] = dct['entity'].to(args.device).type(torch.float32)
                dct['s_and_r'] = dct['s_and_r'].to(args.device)
                dct['target'] = dct['target'].to(args.device)
                label = dct['target'][..., :args.output_dim].squeeze()
                output = model(dct)
                label = label.detach().cpu().numpy()
                output = output.detach().cpu().numpy()

                if len(label.shape) == 1 and args.output_dim!=1:
                    label = np.expand_dims(label, axis=0)
                y_true.append(label)
                y_pred.append(output)
                x_inp.append(dct['s_and_r'].detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        if args.output_dim==1:
            y_true = np.expand_dims(y_true, axis=1)
        y_pred = np.concatenate(y_pred)
        x_inp = np.concatenate(x_inp)
        x_inp = x_inp.reshape((x_inp.shape[0], -1))

        inv_y_pred = np.concatenate((y_pred, x_inp[:, -1:]), axis=1)
        inv_y_true = np.concatenate((y_true, x_inp[:, -1:]), axis=1)

        y_true = scaler.inverse_transform(inv_y_true)
        if args.real_value:
            # y_pred = torch.cat(y_pred, dim=0)
            y_pred = y_pred
        else:
            y_pred = scaler.inverse_transform(inv_y_pred)

        y_pred = y_pred[...,:args.output_dim]
        y_true = y_true[...,:args.output_dim]
        print(y_pred.shape)
        print(y_true.shape)

        logger.info(f"Forecasting results for {args.horizon} days ahead:")
        mae, rmse, mape, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))
        if args.mode == 'train':
            wandb.run.summary['Test_MAE'] = mae
            wandb.run.summary['Test_RMSE'] = rmse
            wandb.run.summary['Test_MAPE'] = mape*100

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
