
python ../slim/converter_mobile_net.py --checkpoint_dir=../models --output_dir=../models/mobile_net_export --image_size=224 --ckptname=model.ckpt-218750 
cd ../models && zip -r mobile_net_export.zip mobile_net_export 

