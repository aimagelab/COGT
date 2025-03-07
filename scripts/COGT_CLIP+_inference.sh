python main.py \
--do_test \
--lr 5e-4 \
--validation_batch_size 16 \
--fp16_enabled \
--num_workers 3 \
--parser roberta \
--resume \
--n_layer 4 \
--n_head 12 \
--n_embd 768 \
--dir_to_save_checkpoint path/to/load/checkpoint