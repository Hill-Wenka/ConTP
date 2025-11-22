import lightning as L
import omegaconf
import torch

from model.ConTP import ConTP
from utils.optim import get_optimizer, get_scheduler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


def compute_label_metrics(preds, labels):
    # preds: [[3], [0]], labels: [[3, 1]], [0]]]
    epoch_metrics = {}
    # compute the weighted avg precision, recall, f1
    mlb = MultiLabelBinarizer()
    mlb.fit(labels + preds)
    y_true = mlb.transform(labels)
    y_pred = mlb.transform(preds)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    samples_precision, samples_recall, samples_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='samples', zero_division=0
    )

    epoch_metrics['weighted_precision'] = weighted_precision
    epoch_metrics['weighted_recall'] = weighted_recall
    epoch_metrics['weighted_f1'] = weighted_f1
    epoch_metrics['samples_precision'] = samples_precision
    epoch_metrics['samples_recall'] = samples_recall
    epoch_metrics['samples_f1'] = samples_f1
    return epoch_metrics


class ConTPModule(L.LightningModule):
    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.model = ConTP(**self.config.hparams)

    def configure_model(self):
        # This hook is called during each of fit/val/test/predict stages in the same process
        pass

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = get_optimizer(self.config.optimizer, params)
        optimizers = [optimizer]
        scheduler = get_scheduler(self.config.scheduler, optimizer)
        schedulers = [scheduler]

        if scheduler is None:
            return optimizers
        else:
            return optimizers, schedulers

    def train_batch_forward(self, batch):
        # print('train_batch_forward')
        data, label = batch  # data: (batch_size, num_sample, feature_dim), label: (batch_size, num_sample, num_class)
        X = self.model(data)  # (batch_size, num_sample, lower_dim)
        loss = self.model.sup_con_hard_loss(X, self.config.dataset.n_pos)
        clusters = self.model.compute_cluster_center(cache_file=self.config.cache_file)  # (num_class, lower_dim)
        X = X[:, 0, :]  # (batch_size, lower_dim)
        label = [x[0] for x in label]  # (batch_size, num_class)

        if self.config.select_labels is not None:
            clusters = clusters[self.config.select_labels, :]
            raw_pred, dist = self.model.find_nearest_cluster(X, clusters, return_dist=True)  # (batch_size, num_class)
            pred = [self.config.select_labels[x] for x in raw_pred]

            # print('X', X.shape)
            # print('raw_pred', raw_pred[:20])
            # print('pred', pred[:20])
            # print('label', label[:20])
            # print('dist', dist[:20])
        else:
            pred, dist = self.model.find_nearest_cluster(X, clusters, return_dist=True)

        cluster_dist = torch.cdist(clusters, clusters, p=2)
        cluster_dist = (cluster_dist.sum() - cluster_dist.diagonal().sum()) / (
                cluster_dist.numel() - cluster_dist.size(0))
        return {'loss': loss, 'label': label, 'pred': pred, 'cluster_dist': cluster_dist}

    def test_batch_forward(self, batch):
        # print('test_batch_forward')
        data, label = batch  # data: (batch_size, feature_dim), label: (batch_size, num_class)
        X = self.model(data)
        clusters = self.model.compute_cluster_center(cache_file=self.config.cache_file)

        if self.config.select_labels is not None:
            clusters = clusters[self.config.select_labels, :]
            raw_pred, dist = self.model.find_nearest_cluster(X, clusters, return_dist=True)
            pred = [self.config.select_labels[x] for x in raw_pred]

            # print('X', X.shape)
            # print('raw_pred', raw_pred[:20])
            # print('pred', pred[:20])
            # print('label', label[:20])
            # print('dist', dist[:20])
        else:
            pred, dist = self.model.find_nearest_cluster(X, clusters, return_dist=True)

        cluster_dist = torch.cdist(clusters, clusters, p=2)
        cluster_dist = (cluster_dist.sum() - cluster_dist.diagonal().sum()) / (
                cluster_dist.numel() - cluster_dist.size(0))
        return {'loss': torch.tensor(torch.nan), 'label': label, 'pred': pred, 'cluster_dist': cluster_dist}

    def compute_batch_metrics(self, batch_outputs):
        batch_metrics = {}
        for key, value in batch_outputs.items():
            if 'loss' in key:
                batch_metrics[key] = value
        return batch_metrics

    def compute_epoch_metrics(self, epoch_outputs):
        epoch_metrics = {}
        epoch_metrics['loss'] = torch.tensor(
            [batch['loss'] for batch in epoch_outputs if not torch.isnan(batch['loss'])]).mean().detach()
        epoch_metrics['cluster_dist'] = torch.tensor([batch['cluster_dist'] for batch in epoch_outputs]).mean().detach()

        preds = []
        for batch in epoch_outputs:
            preds.extend(batch['pred'])
        labels = []
        for batch in epoch_outputs:
            labels.extend(batch['label'])

        if 'substrate' in self.config.cache_file:
            preds = [[x] for x in preds]
        else:
            preds = [[x] for x in preds]
            labels = [[x] for x in labels]

        metrics = compute_label_metrics(preds, labels)
        epoch_metrics.update(metrics)
        return epoch_metrics

    def training_step(self, batch, batch_idx):
        return self.train_batch_forward(batch)

    def validation_step(self, batch, batch_idx):
        return self.test_batch_forward(batch)

    def test_step(self, batch, batch_idx):
        return self.test_batch_forward(batch)

    def on_train_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_metrics = self.compute_batch_metrics(batch_outputs)
        batch_metrics = {'train/' + k + '_step': v for k, v in batch_metrics.items()}
        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.training_step_outputs.append(batch_outputs)

    def on_validation_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_metrics = self.compute_batch_metrics(batch_outputs)
        batch_metrics = {'valid/' + k + '_step': v for k, v in batch_metrics.items()}
        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.validation_step_outputs.append(batch_outputs)

    def on_test_batch_end(self, batch_outputs, batch, batch_idx, **kwargs):
        batch_metrics = self.compute_batch_metrics(batch_outputs)
        batch_metrics = {'test/' + k + '_step': v for k, v in batch_metrics.items()}
        self.log_dict(batch_metrics, on_step=True, on_epoch=False)
        self.test_step_outputs.append(batch_outputs)

    def on_train_epoch_end(self):
        epoch_metrics = self.compute_epoch_metrics(self.training_step_outputs)
        epoch_metrics = {'train/' + k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
        self.training_step_outputs = []

        # import numpy as np
        # concat_embed, cluster_labels = self.model.compute_cluster_center(self.config.cache_file,
        #                                                                  return_sample_embed=True)
        # select_indices = []
        # train_C = []
        # for i in self.config.select_labels:
        #     indices = np.where(cluster_labels == i)[0]
        #     select_indices.extend(indices)
        #     train_C.append(concat_embed[indices].mean(0))
        # train_labels = cluster_labels[select_indices]
        # train_X = concat_embed[select_indices]
        # train_C = torch.stack(train_C, dim=0)
        # train_raw_pred, dist = self.model.find_nearest_cluster(train_X, train_C, return_dist=True)
        # train_pred = np.array([self.config.select_labels[i] for i in train_raw_pred])
        # train_pred = [[y] for y in train_pred]
        # train_labels = [[y] for y in train_labels]
        # train_metrics = compute_label_metrics(train_pred, train_labels)
        # weighted_precision = train_metrics['weighted_precision']
        # weighted_recall = train_metrics['weighted_recall']
        # weighted_f1 = train_metrics['weighted_f1']
        # samples_precision = train_metrics['samples_precision']
        # samples_recall = train_metrics['samples_recall']
        # samples_f1 = train_metrics['samples_f1']

        epoch = self.current_epoch
        step = self.global_step
        weighted_precision = epoch_metrics['train/weighted_precision_epoch']
        weighted_recall = epoch_metrics['train/weighted_recall_epoch']
        weighted_f1 = epoch_metrics['train/weighted_f1_epoch']
        samples_precision = epoch_metrics['train/samples_precision_epoch']
        samples_recall = epoch_metrics['train/samples_recall_epoch']
        samples_f1 = epoch_metrics['train/samples_f1_epoch']
        cd = epoch_metrics['train/cluster_dist_epoch']
        loss = epoch_metrics['train/loss_epoch']
        print(
            f'[Train] epoch: {epoch:02d}, step: {step:03d}, samples_precision: {samples_precision:.2f}, '
            f'samples_recall: {samples_recall:.2f}, samples_f1: {samples_f1:.2f}, '
            f'weighted_precision: {weighted_precision:.2f}, weighted_recall: {weighted_recall:.2f}, '
            f'weighted_f1: {weighted_f1:.2f}, cluster_dist: {cd:.2f}, loss: {loss:.4f}')

    def on_validation_epoch_end(self):
        epoch_metrics = self.compute_epoch_metrics(self.validation_step_outputs)
        epoch_metrics = {'valid/' + k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
        self.validation_step_outputs = []

        epoch = self.current_epoch
        step = self.global_step
        weighted_precision = epoch_metrics['valid/weighted_precision_epoch']
        weighted_recall = epoch_metrics['valid/weighted_recall_epoch']
        weighted_f1 = epoch_metrics['valid/weighted_f1_epoch']
        samples_precision = epoch_metrics['valid/samples_precision_epoch']
        samples_recall = epoch_metrics['valid/samples_recall_epoch']
        samples_f1 = epoch_metrics['valid/samples_f1_epoch']
        cd = epoch_metrics['valid/cluster_dist_epoch']
        print(
            f'[Valid] epoch: {epoch:02d}, step: {step:03d}, samples_precision: {samples_precision:.2f}, '
            f'samples_recall: {samples_recall:.2f}, samples_f1: {samples_f1:.2f}, '
            f'weighted_precision: {weighted_precision:.2f}, weighted_recall: {weighted_recall:.2f}, '
            f'weighted_f1: {weighted_f1:.2f}, cluster_dist: {cd:.2f}')

    def on_test_epoch_end(self):
        epoch_metrics = self.compute_epoch_metrics(self.test_step_outputs)
        epoch_metrics = {'test/' + k + '_epoch': v for k, v in epoch_metrics.items()}
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
        self.test_step_outputs = []

    def predict_step(self, batch, batch_idx, **kwargs):
        return self.test_batch_forward(batch)
