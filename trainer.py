from argparse import Namespace
import glob

import numpy as np
import os
from pathlib import Path
import sys
import torch
from torch import nn
from torch.cuda import amp
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from types import SimpleNamespace
import wandb
import math
from statistics import mean
import einops
from tqdm import tqdm

import logging as python_logging
from datetime import datetime

python_logging.basicConfig(level=python_logging.INFO)

is_debug = 'pydevd' in sys.modules


class Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 training_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 test_data_loaders: dict,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler,
                 device: str,
                 eval_every_fraction_epoch: int,
                 epochs: int,
                 fp16_enabled: bool,
                 debug_steps: int,
                 tokenizer,
                 preprocess_image,
                 wandb_config: Namespace,
                 wandb_run_name: str,
                 wandb_project: str,
                 wandb_entity: str,
                 wandb_mode: str = 'disabled',
                 resume: bool = False,
                 ):
        """
        Trainer class

        :param model:
        :param training_dataloader:
        :param validation_dataloader:
        :param optimizer:
        :param lr_scheduler:
        :param device:
        :param eval_every_fraction_epoch:
        :param epochs:
        :param fp16_enabled:
        :param debug_steps:
        :param wandb_config
        :param wandb_run_name:
        :param wandb_project:
        :param wandb_entity:
        :param wandb_mode:
        :param resume:
        """

        self.model = model
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_data_loaders = test_data_loaders
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.eval_every_fraction_epoch = math.floor(math.floor(len(training_dataloader.dataset) /
                                                               training_dataloader.batch_size) /
                                                    eval_every_fraction_epoch)
        self.epochs = epochs
        self.scaler = amp.GradScaler(enabled=fp16_enabled)
        self.forward_type = torch.float16 if self.scaler.is_enabled() else torch.float32
        self.run_name = wandb_run_name
        self.save_checkpoint_dir = wandb_config.dir_to_save_checkpoint
        self.resume = resume
        self.best_avg_validation_performance = 0
        self.debug_steps = debug_steps
        self.do_train = wandb_config.do_train
        self.do_validation = wandb_config.do_validation
        self.do_test = wandb_config.do_test
        self.args = wandb_config
        self.tokenizer = tokenizer
        self.preprocess_image = preprocess_image
        self.loss_captioner = nn.CrossEntropyLoss(ignore_index=0)
        self.transform_caption = tokenizer["tokenizer"]

        if self.resume:
            wandb_id = self.resume_model()
        if not self.resume or wandb_id is None:
            wandb_id = wandb.util.generate_id()

        self.run = wandb.init(project=wandb_project,
                              entity=wandb_entity,
                              config=wandb_config,
                              id=wandb_id,
                              name=wandb_run_name,
                              resume='allow',
                              mode=wandb_mode)

        self.date = datetime.today().isoformat()
        self.eval_step = 0

    def resume_model(self) -> str:
        """
        """

        # Select path to load
        path_to_load = "/last_checkpoint.pth" if not (self.do_validation or self.do_test) else "/best_checkpoint.pth"
        files = glob.glob(self.save_checkpoint_dir + path_to_load)

        if self.do_train and len(files) == 0:
            python_logging.info("No checkpoint found, starting from scratch..")
            return None

        # Load file and store performances information
        last_checkpoint = torch.load(files[0])
        self.best_avg_validation_performance = last_checkpoint['avg_validation_performance']
        self.model.load_state_dict(last_checkpoint['model_state_dict'])

        python_logging.info(f"Loaded {files[0]}")

        return last_checkpoint['wandb_id']

    def get_check_point_name(self, is_last: bool):
        """
        Return checkpoint name based on specified directory.
        param: is_last when true return last_checkpoint.pth else best_checkpoint.pth
        return:
        """

        python_logging.info("Saving last checkpoint..") if is_last else python_logging.info(
            "Saving best checkpoint on validation..")

        # Create directory if it does not exist
        if not os.path.exists(self.save_checkpoint_dir):
            os.makedirs(self.save_checkpoint_dir)

        # Compute name
        name = "last_checkpoint" if is_last else "best_checkpoint"
        checkpoint_name = Path(self.save_checkpoint_dir, f'{name}.pth')

        return name, checkpoint_name

    def save_checkpoint(self,
                        is_last: bool = True) -> None:
        """
        Save checkpoint
        param: is_last when true save last_checkpoint.pth else best_checkpoint.pth
        """

        name, checkpoint_name = self.get_check_point_name(is_last)

        torch.save({
            'wandb_id': self.run.id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'avg_validation_performance': self.best_avg_validation_performance,
        }, checkpoint_name)

    def train_loop(self, test_compositional: bool) -> None:
        """
        Training loop
        :param: test_compositional specifies whether to compute compositional test metrics
        """
        for _ in range(0, self.epochs):
            self.train()
            if test_compositional:
                test_metrics = self.test_compositional()
                self.log_test_wandb(test_metrics)

    def train(self) -> None:

        self.model.train()
        python_logging.info("Training model..")
        for step, batch in enumerate(self.training_dataloader):
            if is_debug and step > self.debug_steps:
                break

            loss = self.compute_loss(batch)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if step % self.eval_every_fraction_epoch == 0 and step != 0:
                validate_metrics, _ = self.validate()
                self.run.log({"validation/" + k: validate_metrics[k] for k in list(validate_metrics.keys())})

                avg_validation_metric = mean(list(validate_metrics.values()))
                if avg_validation_metric >= self.best_avg_validation_performance:
                    self.best_avg_validation_performance = avg_validation_metric
                    self.save_checkpoint(is_last=False)

                self.save_checkpoint(is_last=True)
                self.model.train()

    def validate(self):
        """
        :return: metrics performances on validation
        """
        self.model.eval()
        python_logging.info("Validate model..")
        for it, batch in enumerate(self.validation_dataloader):
            if is_debug and it > self.debug_steps:
                break

            batch_output = self.prediction_step(batch)
            update_embeddings = {
                "gt_input_ids": np.array(batch_output["gt_input_ids"]),
                "false_input_ids": np.array(batch_output["false_input_ids"]),
                "unique_ids": np.array(batch["unique_ids"])}

            self.validation_dataloader.dataset.update_embeddings(update_embeddings)

        metrics_val = self.validation_dataloader.dataset.compute_metrics()
        self.validation_dataloader.dataset.clean_embedding_dict()

        python_logging.info(f"Validation metric: {metrics_val}")

        return metrics_val

    def arrange_input_for_validation(self,
                                     batch: dict):
        """
        :param batch input dict
        """
        gt_images = batch["gt_images"].to(self.device).to(self.forward_type)
        gt_input_ids = batch["gt_input_ids"].to(self.device)
        false_input_ids = batch["false_input_ids"].to(self.device)
        true_caption_masks = batch["true_caption_mask"].to(self.device)
        gt_likelihood_ids = batch["gt_likelihood_ids"].to(self.device)
        false_captions_masks = batch["false_caption_mask"].to(self.device)
        false_likelihood_ids = batch["false_likelihood_ids"].to(self.device)

        return gt_images, gt_input_ids, false_input_ids, true_caption_masks, false_captions_masks, gt_likelihood_ids, \
               false_likelihood_ids

    def update_test_metrics(self,
                            metrics: dict,
                            dataset_name: str,
                            autoregressive: bool = False) -> dict:
        """
        Update test metrics for the selected dataset
        """

        if dataset_name in ["colorswap"]:
            metrics.update({dataset_name:
                                self.test_data_loaders[dataset_name].dataset.compute_metrics_colorswap()})
        else:
            metrics.update({dataset_name:
                                self.test_data_loaders[dataset_name].dataset.compute_metrics()})
        return metrics

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        :param batch:
        :return:
        """

        inputs = SimpleNamespace(**batch)
        pixel_values = batch['gt_images'].to(device=self.device).to(self.forward_type)
        tokenized_captions = batch['gt_input_ids'].to(device=self.device)
        masks = inputs.true_caption_mask.to(self.device)

        with autocast(dtype=self.forward_type):

            # logits dimension: (B, L, D)
            out = self.model(pixel_values, tokenized_captions, masks)
            shift_logits = out.logits.contiguous()
            shift_labels = batch["ground_truth"].to(torch.int64).to(self.device).contiguous()

            loss = self.loss_captioner(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        self.run.log({"training/training_loss": loss.item()})

        return loss

    def prediction_step(self, batch: dict) -> dict:
        """
        :param batch:
        :return:
        """

        gt_images, gt_input_ids, false_input_ids, true_captions_masks, false_captions_masks, gt_likelihood_ids, \
        false_likelihood_ids = self.arrange_input_for_validation(batch)
        unique_ids = batch["unique_ids"]
        unique_ids_list = [el - min(batch["unique_ids"]).tolist() for el in unique_ids.tolist()]
        embeddings = {}

        with torch.no_grad():
            # Compute outputs embedding
            with autocast(dtype=self.forward_type):
                embeddings_positive = self.model(pixel_values=gt_images,
                                                 captions=gt_input_ids,
                                                 attention_mask=true_captions_masks).logits
                embeddings_negative = self.model(pixel_values=torch.cat([gt_images[int(i)].unsqueeze(dim=0)
                                                                         for i in unique_ids_list], dim=0),
                                                 captions=false_input_ids,
                                                 attention_mask=false_captions_masks).logits

                likelihood_gt = gt_likelihood_ids.to(torch.int64).to(self.device)
                likelihood_neg = false_likelihood_ids.to(torch.int64).to(self.device)

                embeddings.update({"gt_input_ids": compute_likelihood(embeddings_positive,
                                                                      likelihood_gt).cpu()})
                embeddings.update({"false_input_ids": compute_likelihood(embeddings_negative,
                                                                         likelihood_neg).cpu()})
                embeddings.update({"unique_ids": batch["unique_ids"].cpu()})

        return embeddings

    def test_compositional(self) -> dict:
        """
        :return: metrics on test sets
        """

        python_logging.info("Testing model..")

        # Store metrics and embedding dict
        metrics = {}

        self.model.eval()
        # For each dataloader
        for dataset_name in list(self.test_data_loaders.keys()):
            python_logging.info(f"Computing metrics on: {dataset_name}")

            # Cycle over samples
            for it, sample in enumerate(tqdm(self.test_data_loaders[dataset_name],
                                             mininterval=1, total=len(self.test_data_loaders[dataset_name]))):
                if is_debug and it > self.debug_steps:
                    break
                # Read images, true captions and false captions
                images, true_captions, false_captions, true_captions_masks, false_captions_masks, gt_likelihood_ids, \
                false_likelihood_ids = \
                    self.arrange_input_for_validation(sample)
                embeddings = {}

                if "colorswap" in dataset_name:
                    true_captions = torch.cat([einops.repeat(true_captions[i].unsqueeze(dim=0), "1 L -> 2 L")
                                               for i in range(true_captions.shape[0])], dim=0)
                    false_captions = torch.cat([einops.repeat(false_captions[i].unsqueeze(dim=0), "1 L -> 2 L")
                                                for i in range(false_captions.shape[0])], dim=0)
                    true_captions_masks = torch.cat(
                        [einops.repeat(true_captions_masks[i].unsqueeze(dim=0), "1 C H W -> 2 C H W")
                         for i in range(true_captions_masks.shape[0])], dim=0)
                    false_captions_masks = torch.cat(
                        [einops.repeat(false_captions_masks[i].unsqueeze(dim=0), "1 C H W -> 2 C H W")
                         for i in range(false_captions_masks.shape[0])], dim=0)

                    gt_likelihood_ids = torch.cat(
                        [einops.repeat(gt_likelihood_ids[i].unsqueeze(dim=0), "1 L -> 2 L")
                         for i in range(gt_likelihood_ids.shape[0])], dim=0)
                    false_likelihood_ids = torch.cat(
                        [einops.repeat(false_likelihood_ids[i].unsqueeze(dim=0), "1 L -> 2 L")
                         for i in range(false_likelihood_ids.shape[0])], dim=0)

                with torch.no_grad():
                    # Compute outputs embedding
                    if true_captions.shape[0] > images.shape[0]:
                        embeddings_positive = self.model(pixel_values=torch.cat([einops.repeat(
                            images[i].unsqueeze(dim=0), "1 C H W -> N C H W",
                            N=int(true_captions.shape[0] / images.shape[0]))
                            for i in range(images.shape[0])], dim=0),
                            captions=true_captions,
                            attention_mask=true_captions_masks).logits
                    else:
                        embeddings_positive = self.model(pixel_values=images,
                                                         captions=true_captions,
                                                         attention_mask=true_captions_masks).logits

                    ratio = 1

                    if false_captions.shape[0] > images.shape[0]:
                        ratio = int(false_captions.shape[0] / images.shape[0])
                        images = torch.cat([einops.repeat(images[i].unsqueeze(dim=0), "1 C H W -> N C H W", N=ratio)
                                            for i in range(images.shape[0])])

                    embeddings_negative = self.model(pixel_values=images,
                                                     captions=false_captions,
                                                     attention_mask=false_captions_masks).logits

                    # Convention: in gt_images the likelihood with the true caption, in captions the likelihood
                    # with negative
                    gt_images_likelihood = np.expand_dims(np.array(compute_likelihood(embeddings_positive,
                                                                                      gt_likelihood_ids).cpu()),
                                                          axis=1)
                    captions_likelihood = np.array(compute_likelihood(embeddings_negative,
                                                                      false_likelihood_ids).cpu())

                    if len(captions_likelihood.shape) == 1:
                        captions_likelihood = np.expand_dims(captions_likelihood, axis=1)
                    if ratio != 1:
                        captions_likelihood = np.concatenate([np.transpose(captions_likelihood[i:i + ratio])
                                                              for i in
                                                              range(0, captions_likelihood.shape[0], ratio)])

                    embeddings.update({"gt_images": gt_images_likelihood})
                    embeddings.update({"captions": captions_likelihood})

                self.test_data_loaders[dataset_name].dataset.update_embeddings(embeddings)

            # Update metrics dict
            metrics = self.update_test_metrics(metrics, dataset_name)
            # Clean data loader
            self.test_data_loaders[dataset_name].dataset.clean_embedding_dict()

        metrics = process_metrics(metrics)
        log_metrics(metrics)

        return metrics


def compute_likelihood(logits: torch.Tensor, gt_input_ids: torch.Tensor) -> torch.Tensor:
    """
    :param logits:
    :param attention_parser_mask:
    :param gt_input_ids:
    :param likelihood_computation:
    """

    # Mask padding tokens
    logits = torch.nn.functional.softmax(logits, dim=-1)
    logits = torch.gather(logits, 2, einops.repeat(gt_input_ids.unsqueeze(dim=2).to(torch.int64),
                                                   "B L 1 -> B L H",
                                                   H=logits.shape[2]))[:, :, 0]
    logits = torch.log(logits)
    logits[gt_input_ids == 0] = 0
    logits = torch.sum(logits, dim=-1)

    return logits


def process_metrics(metrics: dict) -> dict:
    if "sugar_crepe" in list(metrics.keys()):
        metrics['sugar_crepe']['replace'] = (metrics['sugar_crepe']['fine_grained_results']['replace_obj'] +
                                             metrics['sugar_crepe']['fine_grained_results']['replace_rel'] +
                                             metrics['sugar_crepe']['fine_grained_results']['replace_att']) / 3
        metrics['sugar_crepe']['swap'] = (metrics['sugar_crepe']['fine_grained_results']['swap_obj'] +
                                          metrics['sugar_crepe']['fine_grained_results']['swap_att']) / 2
        metrics['sugar_crepe']['add'] = (metrics['sugar_crepe']['fine_grained_results']['add_obj'] +
                                         metrics['sugar_crepe']['fine_grained_results']['add_att']) / 2
        del metrics['sugar_crepe']['fine_grained_results']

    if 'visual_genome_relation' in list(metrics.keys()):
        del metrics['visual_genome_relation']['fine_grained_results']
    if 'visual_genome_attribution' in list(metrics.keys()):
        del metrics['visual_genome_attribution']['fine_grained_results']

    return metrics


def log_metrics(metrics: dict):
    for dataset in metrics:
        python_logging.info(f"\n------------------\nMetrics for {dataset}:")
        metrics_formatted = metrics_format(metrics[dataset])
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        metrics_to_print = metrics_formatted.keys()
        for key in sorted(metrics_formatted.keys()):
            python_logging.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")
        if 'avg_accuracy' not in metrics_to_print and 'avg_accuracy' in metrics[dataset]:
            python_logging.info(f"  {'avg_accuracy': <{k_width}} = {metrics[dataset]['avg_accuracy']:>{v_width}}")
        if 'top_1_recall_img2text' not in metrics_to_print and 'top_1_recall_img2text' in metrics[dataset]:
            if 'top_1_recall_img2text' not in metrics_to_print and 'top_1_recall_img2text' in metrics[dataset]:
                python_logging.info(
                    f"  {'top_1_recall_img2text': <{k_width}} = {metrics[dataset]['top_1_recall_img2text']:>{v_width}}")


def metrics_format(metrics):
    """
    Reformat Trainer metrics values to a human-readable format

    Args:
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict

    Returns:
        metrics (`Dict[str, float]`): The reformatted metrics
    """

    if 'fine_grained_results' in metrics:
        metrics_copy = dict(sorted(metrics['fine_grained_results'].copy().items()))
    else:
        metrics_copy = dict(sorted(metrics.copy().items()))

    for k, v in metrics_copy.items():
        metrics_copy[k] = round(v, 4)

    return metrics_copy
