python main.py \
--do_train \
--lr 5e-4 \
--training_batch_size 128 \
--validation_batch_size 16 \
--fp16_enabled \
--shuffle \
--eval_every_fraction_epoch 16 \
--wandb_run_name run_name \
--num_workers 3 \
--wandb_mode online \
--parser roberta \
--dir_to_save_checkpoint path/to/save/checkpoint