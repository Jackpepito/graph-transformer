export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 2 \
--data_path "/content/gdrive/MyDrive/graph-project/graph-transformer/" \
--train_set "/content/gdrive/MyDrive/graph-project/OBR/train.txt" \
--val_set "/content/gdrive/MyDrive/graph-project/OBR/val.txt" \
--model_path "/content/gdrive/MyDrive/graph-project/graph-transformer/graph-transformer/saved_models/" \
--log_path "/content/gdrive/MyDrive/graph-project/graph-transformer/graph-transformer/runs/" \
--task_name "GraphCAM" \
--batch_size 8 \
--train \
--log_interval_local 6 \
