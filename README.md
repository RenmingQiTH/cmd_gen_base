# Instruction
## Requirement 
1. Tensorflow 1.6 with GPU.
2. Anaconda 
3. Download cmd_gen.zip and unzip it as  `unzip cmd_gen_rcmp.zip`

## Prepare dataset
1. ``` cd cmd_gen_rcmp```
2. This is a foler `data` under `cmd_gen_rcmp`.
 Copy adult images to `data/rcmp/adult`and csam data to `data/rcmp/csam`. 
 3. Now create the tfrecords.  Ensure the current working directory is under `cmd_gen_rcmp`. 
 ```
 python create_tfrecords/create_tfrecord.py --dataset_dir=./data --tfrecord_filename=rcmp
 ```
## Train and Evaluate 
1. First go to cmd_gen directory as
```  
cd ./cmd_gen
```
2. Run 
```
python config_writer.py
```
3. Done. 
## Response 
1. There will be some new folders under `./cmd/gen`, which are running results. Please send us back them.  You may not need to send us back the folders whose name start with **useless** since they are too huge. However, please keep them. We might need them in the future. 
2. There will be a file named `labels.txt` under `data` folder. Please also send it back to us. 


