import os
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from math import sqrt

from model import TopicClassCNN 
from dataset import Vocabulary
from dataloader import TopicClassDataLoader
from utils.metric import MetricTracker, accuracy
from utils.summary import TensorboardWriter


class Trainer(object):
    def __init__(self, config):
        self.logger = logging.getLogger("Training")
        self.config = config
        self.start_epoch = 1
        self.monitor = self.config.monitor
        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = self.config.early_stop
        self.prepare_device()

        self.logger.info("Creating tensorboard writer...")
        self.writer = TensorboardWriter(
            log_dir=self.config.summary_dir, logger=self.logger, enabled=True)

        self.logger.info("Creating data loaders...")
        self.build_data_loader()

        self.logger.info("Creating model architecture...")
        self.build_model()

        self.logger.info("Creating optimizers...")
        self.build_optimizer()

        self.logger.info("Creating losses...")
        self.build_loss()

        self.logger.info("Creating metric trackers...")
        self.build_metrics()

        self.logger.info("Creating checkpoints...")
        self.load_checkpoint(self.config.checkpoint, self.config.resume_epoch)

        self.logger.info("Check parallelism...")
        self.parallelism()

    def build_model(self):
        if self.config.pretrain_emb:
            # load pretrained word2vec into model
            self.logger.info("load pretrained embedding")
            weight_matrix = self.vocabulary.get_word2vec_weight(model_name='fasttext-crawl-2M', emb_size=self.config.emb_size)
            weight_tensor = torch.FloatTensor(weight_matrix)
        else:
            weight_tensor = None

        self.model = TopicClassCNN(
            vocab_size=self.config.vocab_size, # add 1 for <pad>
            emb_size=self.config.emb_size,
            dropout=self.config.dropout,
            kernel_sizes=self.config.kernel_sizes,
            num_feat_maps=self.config.num_feat_maps,
            num_classes=self.config.num_classes,
            weight_tensor=weight_tensor,
            freeze=self.config.static)
        
        self.logger.info("embedding reqeuire grad: {}".format(self.model.embedding.weight.requires_grad))
        

    def build_data_loader(self):
        self.vocabulary = Vocabulary()
        self.config.vocab_size = self.vocabulary.get_num_words()
        assert self.config.num_classes == self.vocabulary.get_num_topics(), "class number doesn't match topic number"

        self.train_loader = TopicClassDataLoader(
            vocabulary=self.vocabulary,
            split='train',
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )

        self.valid_loader = TopicClassDataLoader(
            vocabulary=self.vocabulary,
            split='valid',
            batch_size=1,
            num_workers=self.config.num_workers)

    def build_optimizer(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.config.lr, weight_decay=self.config.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.exp_lr_gamma)

        self.logger.info("embedding reqeuire grad: {}".format(
            self.model.embedding.weight.requires_grad))

    def build_loss(self):
        self.cls_loss = nn.CrossEntropyLoss()
        self.cls_loss.to(self.device)

    def build_metrics(self):
        loss_tags = ['loss', 'acc_1', 'acc_5']
        self.train_metrics = MetricTracker(*loss_tags, writer=self.writer)
        self.val_metrics = MetricTracker(*loss_tags, writer=self.writer)

    def train(self):
        not_improved_count = 0
        self.logger.info("Starting training...")

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            result = self.train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode ==
                                'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            self.save_checkpoint('latest', save_best=False)
            if epoch % self.config.save_period == 0:
                self.save_checkpoint(epoch, save_best=best)

            self.config.resume_epoch = epoch

    def train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (sentences, gt_topics, lengths, masks) in enumerate(self.train_loader):
            sentences = sentences.to(self.device)
            gt_topics = gt_topics.to(self.device)
            masks = masks.to(self.device)

            pred_topics = self.model(sentences, masks)
            loss = self.cls_loss(pred_topics, gt_topics)
            acc = accuracy(pred_topics, gt_topics)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # add loss summary when update generator to save memory
            self.writer.set_step((epoch - 1) * len(self.train_loader) + batch_idx, mode='train')
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('acc_1', acc[0].item())
            self.train_metrics.update('acc_5', acc[1].item())

            # log on console
            if batch_idx % self.config.summary_step == 0:
                self.logger.info('Train Epoch: {} {} Loss:{:.4f}, Acc_1: {:.2f}, Acc_5: {:.2f}]'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    acc[0].item(), acc[1].item()))

        self.lr_scheduler.step()
        log = self.train_metrics.result()
        val_log = self.valid_epoch(epoch)
        log.update(**{'val_' + k: v for k, v in val_log.items()})
        return log

    def valid_epoch(self, epoch):
        self.model.eval()
        val_loss = []
        val_acc = []
        val_acc5 = []
        with torch.no_grad():
            for batch_idx, (sentences, gt_topics, lengths, masks) in enumerate(self.valid_loader):
                sentences, gt_topics = sentences.to(self.device), gt_topics.to(self.device)
                masks = masks.to(self.device)

                pred_topics=self.model(sentences, masks)
                loss=self.cls_loss(pred_topics, gt_topics)
                acc = accuracy(pred_topics, gt_topics)

                val_loss.append(loss.item())
                val_acc.append(acc[0].item())
                val_acc5.append(acc[1].item())

        self.writer.set_step(epoch, mode='val')
        self.val_metrics.update('loss', np.mean(val_loss))
        self.val_metrics.update('acc_1', np.mean(val_acc))
        self.val_metrics.update('acc_5', np.mean(val_acc5))

        return self.val_metrics.result()

    def save_checkpoint(self, epoch, save_best):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        filename = 'epoch_{}.pth'.format(epoch)
        torch.save(state, os.path.join(self.config.checkpoint_dir, filename))
        if save_best:
            best_path = os.path.join(
                self.config.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def load_checkpoint(self, checkpoint_dir=None, epoch=None):
        if checkpoint_dir is None:
            self.logger.info("Training from scratch...")
            self.model.to(self.device)
            self.start_epoch = 1
            return

        self.logger.info(
            "Loading checkpoints from {}...".format(checkpoint_dir))
        self.start_epoch = epoch + 1
        self.logger.info("Continuing training from epoch {}...".format(epoch))
        filename = 'epoch_{}.pth'.format(epoch)
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        model_to_load = {
            k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(model_to_load)
        self.model.to(self.device)
        if self.config.mode == 'train':
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def prepare_device(self):
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device("cuda:0")
            self.logger.info("Training will be conducted on GPU")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Training will be conducted on CPU")

        n_gpu = torch.cuda.device_count()
        n_gpu_use = self.config.ngpu

        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
            self.config.ngpu = n_gpu_use
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        self.device_ids = list(range(n_gpu_use))

    def parallelism(self):
        if len(self.device_ids) > 1:
            self.logger.info("Using {} GPUs...".format(len(self.device_ids)))
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.device_ids)
        else:
            if self.cuda:
                self.logger.info(
                    "Using only 1 GPU and do not parallelize the models...")
            else:
                self.logger.info("Using CPU...")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = len(self.train_loader)
        return base.format(current, total, 100.0 * current / total)
