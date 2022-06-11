export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 2 \
--data_path "/content/gdrive/MyDrive/graph-project/graph-transformer/" \
--val_set "/content/gdrive/MyDrive/graph-project/graph-transformer/graphs/test.txt" \
--model_path "/content/gdrive/MyDrive/graph-project/graph-transformer/graph-transformer/runs/saved_models/" \
--log_path "/content/gdrive/MyDrive/graph-project/graph-transformer/graph-transformer/runs/" \
--task_name "GraphCAM" \
--batch_size 1 \
--test \
--log_interval_local 6 \
--resume "/content/gdrive/MyDrive/graph-project/graph-transformer/graph-transformer/runs/saved_models/GraphCAM.pth"
