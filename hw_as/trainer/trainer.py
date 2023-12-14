import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from hw_as.base import BaseTrainer
from hw_as.utils import inf_loop, MetricTracker
from hw_as.metric import calculate_tDCF_EER


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["audios", "labels"]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch), start=1
        ):
            if 'error' in batch:
                continue

            if batch_idx > self.len_epoch:
                break
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0 or batch_idx == 1:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx - 1)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        if (epoch + 1) % 10 == 0:
            for part, dataloader in self.evaluation_dataloaders.items():
                val_log = self._evaluation_epoch(epoch, part, dataloader)
                for name, value in val_log.items():
                    log.update(**{name: value})
                    self.writer.add_scalar(
                        name, value
                    )

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        if is_train:
            self.optimizer.zero_grad()

        out = self.model(batch['audios'])
        batch.update(out)

        if is_train:
            losses = self.criterion(**batch)
            batch.update(losses)
            losses["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            for loss_name, loss_value in losses.items():
                metrics.update(loss_name, loss_value.item())
            for met in self.metrics:
                metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.move_batch_to_device(batch, self.device)
                out = self.model(batch['audios'])
                batch.update(out)

                if batch_idx == 0:
                    descriptor = 'w'
                else:
                    descriptor = 'a'

                with open('cm_scores_eval.txt', descriptor) as f:
                    for label, attack, logit in zip(batch['labels'].cpu(), batch['attack'], batch['logits'].cpu()):
                        if label == 1:
                            source = 'bonafide'
                        else:
                            source = 'spoof'
                        f.write("_" + " " + attack + " " + source + " " + str(logit[1].item()))
                        f.write('\n')

            asv_scores_file = '/kaggle/input/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt'
            eer_cm, min_tDCF = calculate_tDCF_EER('cm_scores_eval.txt',
                                                  asv_scores_file,
                                                  'output.txt')

        self.writer.set_step(epoch * self.len_epoch, part)
        return {
            'eer_CM': eer_cm,
            'min_tDCF': min_tDCF
        }

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))