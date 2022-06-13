export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 2 \
--data_path "/content/gdrive/MyDrive/graph-transformer/graphs/OBR/" \
--train_set "/content/gdrive/MyDrive/graph-transformer/OBR/train.txt" \
--val_set "/content/gdrive/MyDrive/graph-transformer/OBR/val.txt" \
--model_path "/content/gdrive/MyDrive/graph-transformer/graph-transformer/saved_models/" \
--log_path "/content/gdrive/MyDrive/graph-transformer/graph-transformer/runs/" \
--task_name "GraphCAM" \
--batch_size 2 \
--train \
--log_interval_local 6 \
