import json 
import os
import copy
import argparse
from gen_util import * 

path_v2 = "../models/pretrained/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt"
path_v3 = "../models/pretrained/inception_v3/inception_v3.ckpt"
path_mobile = "../models/pretrained/mobilenet_v1/mobilenet_v1_1.0_224.ckpt"
DATASET_DIR="../data"
def main(istest):
    def gen_validation_and_train_eval(eval_config,csv_path_middle,csv_path_final):
        eval_config['labels_offset']="0"
        eval_config_val_middle = copy.deepcopy(eval_config)
        eval_config_val_final = copy.deepcopy(eval_config)
        eval_config_train_middle = copy.deepcopy(eval_config)
        eval_config_train_final = copy.deepcopy(eval_config)
        eval_config_val_middle['dataset_split_name'] = "validation"
        eval_config_val_middle['csv_name'] = csv_path_middle + "_val.csv"
        eval_config_val_final['dataset_split_name'] = "validation"
        eval_config_val_final['csv_name'] = csv_path_final + "_val.csv"
        eval_config_train_middle['dataset_split_name']="train"
        eval_config_train_middle['csv_name'] = csv_path_middle + "_train.csv"
        eval_config_train_final['dataset_split_name']="train"
        eval_config_train_final['csv_name'] = csv_path_final + "_train.csv"
        return eval_config_val_middle,eval_config_train_middle,eval_config_val_final,eval_config_train_final
    def gen_forzen_and_unfrozen_train(train_config):
        train_config["labels_offset"]="0"

        train_config_frozen = copy.deepcopy(train_config)
        train_config_frozen["max_number_of_steps"] =str(31250)
        train_config_unfrozen = copy.deepcopy(train_config)
        train_config_unfrozen['checkpoint_path'] = train_config_frozen['train_dir']
        train_config_unfrozen["learning_rate"]="0.0001"
        train_config_unfrozen["max_number_of_steps"] =str(218750)
        return train_config_frozen, train_config_unfrozen


    # DATASET_DIR="../data"
    # folder_name = "inception_resnet_v2_epoch_70"
    # config_writer = Config_Writer(folder_name)
    # train_folder =folder_name
    # ck_path , train_p, eval_p,csv_path_middle = path_gen(path_v2,train_folder=train_folder, csv_name_prefix ="middle")
    # ck_path , train_p, eval_p,csv_path_final = path_gen(path_v2,train_folder=train_folder, csv_name_prefix = "final")

    # config_gen = ConfigFactory()
    # train_config = config_gen.gen_fn("train:inception_resnet_v2")
    # train_config = train_config(ck_path,DATASET_DIR,train_p)

    # config_gen = ConfigFactory()
    # eval_config = config_gen.gen_fn("eval:inception_resnet_v2")
    # eval_config = eval_config(train_p,DATASET_DIR,eval_p)

    # evm,etm,evf,etf = gen_validation_and_train_eval(eval_config,csv_path_middle,csv_path_final)
    # tfz, tuf = gen_forzen_and_unfrozen_train(train_config)
    # task_seq = [tfz,evm,etm,tuf,evf,etf]
    # task_describe = {"task_name":folder_name,"task_seq":task_seq}

    # config_writer = Config_Writer(os.path.join(".",folder_name))
    # config_writer.write("inception_resnet_v2.json",task_describe)
    # runner = SeqRunner(task_dict=task_describe)
    # if not istest:
    #     try:
    #         runner.run()
    #     except:
    #         pass 


    # DATASET_DIR="../data"
    # folder_name = "inception_v3_epoch_70"
    # config_writer = Config_Writer(folder_name)
    # train_folder =folder_name
    # ck_path , train_p, eval_p,csv_path_middle = path_gen(path_v3,train_folder=train_folder, csv_name_prefix ="middle")
    # ck_path , train_p, eval_p,csv_path_final = path_gen(path_v3,train_folder=train_folder, csv_name_prefix = "final")

    # config_gen = ConfigFactory()
    # train_config = config_gen.gen_fn("train:inception_v3")
    # train_config = train_config(ck_path,DATASET_DIR,train_p)

    # config_gen = ConfigFactory()
    # eval_config = config_gen.gen_fn("eval:inception_v3")
    # eval_config = eval_config(train_p,DATASET_DIR,eval_p)

    # evm,etm,evf,etf = gen_validation_and_train_eval(eval_config,csv_path_middle,csv_path_final)
    # tfz, tuf = gen_forzen_and_unfrozen_train(train_config)
    # task_seq = [tfz,evm,etm,tuf,evf,etf]
    # task_describe = {"task_name":folder_name,"task_seq":task_seq}


    # config_writer = Config_Writer(os.path.join(".",folder_name))
    # config_writer.write("inception_v3.json",task_describe)
    # runner = SeqRunner(task_dict=task_describe)
    # if not istest:
    #     try:
    #         runner.run()
    #     except:
    #         pass 


    # DATASET_DIR="../data"
    # folder_name = "mobile_net_v1_epoch_70"
    # config_writer = Config_Writer(folder_name)
    # train_folder =folder_name
    # ck_path , train_p, eval_p,csv_path_middle = path_gen(path_mobile,train_folder=train_folder, csv_name_prefix ="middle")
    # ck_path , train_p, eval_p,csv_path_final = path_gen(path_mobile,train_folder=train_folder, csv_name_prefix = "final")

    # config_gen = ConfigFactory()
    # train_config = config_gen.gen_fn("train:mobile_net_v1")
    # train_config = train_config(ck_path,DATASET_DIR,train_p)

    # config_gen = ConfigFactory()
    # eval_config = config_gen.gen_fn("eval:mobile_net_v1")
    # eval_config = eval_config(train_p,DATASET_DIR,eval_p)

    # evm,etm,evf,etf = gen_validation_and_train_eval(eval_config,csv_path_middle,csv_path_final)
    # tfz, tuf = gen_forzen_and_unfrozen_train(train_config)
    # task_seq = [tfz,evm,etm,tuf,evf,etf]
    # task_describe = {"task_name":folder_name,"task_seq":task_seq}


    # config_writer = Config_Writer(os.path.join(".",folder_name))
    # config_writer.write("mobile.json",task_describe)
    # runner = SeqRunner(task_dict=task_describe)
    # if not istest:
    #     try:
    #         runner.run()
    #     except:
    #         pass 

    #--------------------------------------------------------------------------------#
    DATASET_DIR="../data"
    folder_name = "inception_resnet_v2_epoch_140"
    config_writer = Config_Writer(folder_name)
    train_folder =folder_name
    ck_path , train_p, eval_p,csv_path_middle = path_gen(path_v2,train_folder=train_folder, csv_name_prefix ="middle")
    ck_path , train_p, eval_p,csv_path_final = path_gen(path_v2,train_folder=train_folder, csv_name_prefix = "final")

    config_gen = ConfigFactory()
    train_config = config_gen.gen_fn("train:inception_resnet_v2")
    train_config = train_config(ck_path,DATASET_DIR,train_p)

    config_gen = ConfigFactory()
    eval_config = config_gen.gen_fn("eval:inception_resnet_v2")
    eval_config = eval_config(train_p,DATASET_DIR,eval_p)

    evm,etm,evf,etf = gen_validation_and_train_eval(eval_config,csv_path_middle,csv_path_final)
    tfz, tuf = gen_forzen_and_unfrozen_train(train_config)
    tuf["max_number_of_steps"] = str(218750 * 2 )
    task_seq = [tfz,evm,etm,tuf,evf,etf]
    task_describe = {"task_name":folder_name,"task_seq":task_seq}

    config_writer = Config_Writer(os.path.join(".",folder_name))
    config_writer.write("inception_resnet_v2.json",task_describe)
    runner = SeqRunner(task_dict=task_describe)
    if not istest:
        try:
            runner.run()
        except:
            pass 

    DATASET_DIR="../data"
    folder_name = "inception_v3_epoch_140"
    config_writer = Config_Writer(folder_name)
    train_folder =folder_name
    ck_path , train_p, eval_p,csv_path_middle = path_gen(path_v3,train_folder=train_folder, csv_name_prefix ="middle")
    ck_path , train_p, eval_p,csv_path_final = path_gen(path_v3,train_folder=train_folder, csv_name_prefix = "final")

    config_gen = ConfigFactory()
    train_config = config_gen.gen_fn("train:inception_v3")
    train_config = train_config(ck_path,DATASET_DIR,train_p)

    config_gen = ConfigFactory()
    eval_config = config_gen.gen_fn("eval:inception_v3")
    eval_config = eval_config(train_p,DATASET_DIR,eval_p)

    evm,etm,evf,etf = gen_validation_and_train_eval(eval_config,csv_path_middle,csv_path_final)
    tfz, tuf = gen_forzen_and_unfrozen_train(train_config)
    tuf["max_number_of_steps"] = str(218750 * 2 )

    task_seq = [tfz,evm,etm,tuf,evf,etf]
    task_describe = {"task_name":folder_name,"task_seq":task_seq}


    config_writer = Config_Writer(os.path.join(".",folder_name))
    config_writer.write("inception_v3.json",task_describe)
    runner = SeqRunner(task_dict=task_describe)
    if not istest:
        try:
            runner.run()
        except:
            pass 


    DATASET_DIR="../data"
    folder_name = "mobile_net_v1_epoch_140"
    config_writer = Config_Writer(folder_name)
    train_folder =folder_name
    ck_path , train_p, eval_p,csv_path_middle = path_gen(path_mobile,train_folder=train_folder, csv_name_prefix ="middle")
    ck_path , train_p, eval_p,csv_path_final = path_gen(path_mobile,train_folder=train_folder, csv_name_prefix = "final")

    config_gen = ConfigFactory()
    train_config = config_gen.gen_fn("train:mobile_net_v1")
    train_config = train_config(ck_path,DATASET_DIR,train_p)

    config_gen = ConfigFactory()
    eval_config = config_gen.gen_fn("eval:mobile_net_v1")
    eval_config = eval_config(train_p,DATASET_DIR,eval_p)

    evm,etm,evf,etf = gen_validation_and_train_eval(eval_config,csv_path_middle,csv_path_final)
    tfz, tuf = gen_forzen_and_unfrozen_train(train_config)
    tuf["max_number_of_steps"] = str(218750 * 2 )

    task_seq = [tfz,evm,etm,tuf,evf,etf]
    task_describe = {"task_name":folder_name,"task_seq":task_seq}


    config_writer = Config_Writer(os.path.join(".",folder_name))
    config_writer.write("mobile.json",task_describe)
    runner = SeqRunner(task_dict=task_describe)
    if not istest:
        try:
            runner.run()
        except:
            pass 
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="increase output verbosity")
    args = parser.parse_args()
    istest = False
    if args.test:
        istest = True
    main(istest)