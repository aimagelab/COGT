import argparse
from torch.utils.data import DataLoader
from dataset import get_compositional_data_loader, \
    TrainingSet, ValidationSet, collate_fn
import torch
import sys
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from paths import PATH, TEST_PATH, get_xvlm_transform
import logging as python_logging
from cap_models import CapWrapper
from models import open_clip_lora
import yaml

from trainer import Trainer
from xvlm_dir.xvlm import XVLMBase

python_logging.basicConfig(level=python_logging.INFO)

is_debug = 'pydevd' in sys.modules


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_image_dir_path', type=str, default=PATH['coco_images_negCLIP'])
    parser.add_argument('--training_metadata_path', type=str, default=PATH['coco_training_meta_negCLIP'])
    parser.add_argument('--modality', type=str, default="captioning")
    parser.add_argument('--training_recap', action='store_true')
    parser.add_argument('--training_recap_plus_coco', action='store_true')

    # Wandb
    parser.add_argument('--wandb_run_name', type=str)
    parser.add_argument('--wandb_entity', type=str, default='wandb_entity')
    parser.add_argument('--wandb_project', type=str, default='wandb_project')
    parser.add_argument('--wandb_mode', type=str, default='disabled')

    # Model
    parser.add_argument('--cap_visual_backbone', type=str, default='ViT-B-32')
    parser.add_argument('--xvlm', action='store_true')
    parser.add_argument('--instruct_blip', action='store_true')
    parser.add_argument('--pretrained_path_visual_backbone', type=str, default="laion2b_s34b_b79k")
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=512)

    # Training
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_validation', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--fp16_enabled', action='store_true')
    parser.add_argument('--optimizer', default=torch.optim.Adam)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--learning_rate_scheduler', default=torch.optim.lr_scheduler.CosineAnnealingLR)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--eval_every_fraction_epoch', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dir_to_save_checkpoint', type=str, default='output_dir')
    parser.add_argument('--test_compositional', action='store_true')
    parser.add_argument('--parser', type=str, default="roberta")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_datasets', type=str, nargs='+', default='coco_training')
    parser.add_argument('--validation_datasets', type=str, nargs='+', default='coco_validation_dict')
    parser.add_argument('--test_datasets', nargs='+', default=[])
    parser.add_argument('--training_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=4)

    # Validation
    parser.add_argument('--validation_batch_size', type=int, default=4)
    parser.add_argument('--validation_image_dir_path', type=str, default=PATH['coco_images_negCLIP'])
    parser.add_argument('--validation_metadata_path', type=str, default=PATH['coco_val_meta_negCLIP'])

    # Debug
    parser.add_argument("--debug_steps", type=int, default=5)

    custom_args = parser.parse_args()

    # Set seed
    torch.manual_seed(custom_args.seed)

    tokenizer = {"name": "open_clip",
                 "tokenizer": open_clip_lora.get_tokenizer(custom_args.cap_visual_backbone)}
    tokenizer_q_former = None

    if custom_args.xvlm:
        with open(PATH['config_xvlm']) as f:
            config = yaml.safe_load(f)
            config["vision_config"] = PATH["config_swin_xvlm"]

        preprocess_image = get_xvlm_transform(config)
        visual_backbone = XVLMBase(config=config, load_vision_params=True)
        visual_backbone.load_pretrained(ckpt_rpath=PATH["xvlm_weights"], config=config)
        visual_backbone = visual_backbone.vision_encoder
    elif custom_args.instruct_blip:
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        preprocess_image = processor.image_processor
        tokenizer_q_former = processor.tokenizer
        visual_backbone = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        del visual_backbone.language_model
        del visual_backbone.language_projection

    else:
        visual_backbone, _, preprocess_image = open_clip_lora.create_model_and_transforms(
            custom_args.cap_visual_backbone,
            device=custom_args.device,
            pretrained=custom_args.pretrained_path_visual_backbone,
        )

    model = CapWrapper(visual_backbone=visual_backbone,
                       tokenizer=tokenizer['tokenizer'],
                       tokenizer_q_former=tokenizer_q_former,
                       args=custom_args)

    training_dataset = TrainingSet
    image_dir_path = custom_args.training_image_dir_path
    metadata_path = custom_args.training_metadata_path
    validation_dataset = ValidationSet

    # Initialize dataloaders
    training_dataloader = DataLoader(
        dataset=training_dataset(image_dir_path=image_dir_path,
                                 metadata_path=metadata_path,
                                 preprocess_image=preprocess_image,
                                 tokenizer=tokenizer,
                                 parser=custom_args.parser,
                                 n_head=custom_args.n_head,
                                 args=custom_args),
        shuffle=custom_args.shuffle,
        batch_size=custom_args.training_batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=custom_args.num_workers)

    validation_dataloader = DataLoader(validation_dataset(image_dir_path=custom_args.validation_image_dir_path,
                                                          metadata_path=custom_args.validation_metadata_path,
                                                          preprocess_image=preprocess_image,
                                                          tokenizer=tokenizer,
                                                          parser=custom_args.parser,
                                                          n_head=custom_args.n_head,
                                                          args=custom_args),
                                       shuffle=False,
                                       batch_size=custom_args.validation_batch_size,
                                       collate_fn=collate_fn,
                                       num_workers=custom_args.num_workers)

    test_datasets = custom_args.test_datasets if custom_args.test_datasets else list(TEST_PATH.keys())
    compositional_dataset_dict = {
        dataset_name: get_compositional_data_loader(image_dir_path=TEST_PATH[dataset_name]["images"],
                                                    metadata_path=TEST_PATH[dataset_name]["metadata"],
                                                    preprocess_image=preprocess_image,
                                                    tokenizer=tokenizer,
                                                    batch_size=custom_args.validation_batch_size,
                                                    num_workers=custom_args.num_workers,
                                                    parser=custom_args.parser,
                                                    n_head=custom_args.n_head,
                                                    args=custom_args) for
        dataset_name in test_datasets}

    optimizer = custom_args.optimizer(model.parameters(), lr=custom_args.lr)

    lr_scheduler = custom_args.learning_rate_scheduler(optimizer,
                                                       custom_args.warmup_steps)

    trainer = Trainer
    trainer = trainer(model=model,
                      training_dataloader=training_dataloader,
                      validation_dataloader=validation_dataloader,
                      test_data_loaders=compositional_dataset_dict,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      device=custom_args.device,
                      eval_every_fraction_epoch=custom_args.eval_every_fraction_epoch,
                      epochs=custom_args.epochs,
                      fp16_enabled=custom_args.fp16_enabled,
                      wandb_config=custom_args,
                      tokenizer=tokenizer,
                      preprocess_image=preprocess_image,
                      wandb_run_name=custom_args.wandb_run_name,
                      wandb_project=custom_args.wandb_project,
                      wandb_entity=custom_args.wandb_entity,
                      wandb_mode=custom_args.wandb_mode,
                      resume=custom_args.resume,
                      debug_steps=custom_args.debug_steps)

    if custom_args.do_train:
        trainer.train_loop(custom_args.test_compositional)

    if custom_args.do_validation:
        trainer.validate()

    if custom_args.do_test:
        trainer.test_compositional()


if __name__ == "__main__":
    main()
