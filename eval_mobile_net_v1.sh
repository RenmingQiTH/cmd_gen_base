python ./slim/eval_image_classifier_summary.py \
--csv_name=eval_mobile_output.csv \
 --model_name=mobilenet_v1  \ 
 --dataset_dir=./data \
 --checkpoint_path = ./models/model.ckpt-218750\
 --labels_offset=0 \
 --dataset_split_name = validation\
 --dataset_name=rcmp\