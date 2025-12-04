python train.py \
  --train_data_dir ./training_dataset/images \
  --caption_dir ./training_dataset \
  --train_batch_size 4 \
  --max_train_steps 2000 \
  --learning_rate 1e-5 \
  --use_wandb  # Optional for logging


<br>
  # Command line
python generate.py --prompt "your description here"

# Web interface
python generate.py --gradio
