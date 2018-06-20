import os 
import copy 
import json
from subprocess import Popen, PIPE
class _gen_flags:
    def __init__(self):
        pass

gen_flags = _gen_flags()
gen_flags.eval_path = "../eval_image_classifier_summary.py"
gen_flags.train_path = "../train_image_classifier.py"
script_map = {"eval":gen_flags.eval_path, "train": gen_flags.train_path}
script_map = {"eval":gen_flags.eval_path, "train": gen_flags.train_path}

class GenCmd:
    def __init__(self, config_dict):
        self._config_dict = config_dict
    def _gen_cmd(self):
        str_cmd = ""
        for k, v in self._config_dict.items():
            if k!="method":
                str_cmd  =" " +  str_cmd + "--" +  k +"="+ v + " "
        return "python " + script_map[self._config_dict["method"]] + " " + str_cmd
    
    def _gen_cmd_list(self):
        list_cmd  = []
        list_cmd.append("python")
        list_cmd.append(script_map[self._config_dict["method"]])
        for k, v in self._config_dict.items():
            if k!="method":
                list_cmd.append("--"+k)
                list_cmd.append(v)
        return list_cmd
    def execute_cmd(self):
        cmd = self._gen_cmd_list()
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        rc = p.returncode
        #return bytes str. output : stdout, err: stderr, rc: return code 
        return output,err, rc 

class SeqRunner:
    def __init__(self,task_dict):
        self.task_name = task_dict["task_name"]
#         self.target_dir = task_json["target_dir"]
        self.task_seq = task_dict["task_seq"]
        self.task_dict = copy.deepcopy(task_dict)
    def run(self):
        
        ind = 0 
        task_seq = copy.deepcopy(self.task_seq)
        for task in task_seq:
            print("_______________")
            print(task)
            print("_______________")
            describe = task
            cmd = GenCmd(task)
            normal_output, err_output, rcode = cmd.execute_cmd()
            describe["std_out"]=normal_output.decode("utf8")
            describe["err_output"]=err_output.decode("utf8")
            describe["return_code"]=str(rcode)
            describe["deep_copy_test"]="unexpected"
            self.task_dict["task_seq"][ind ] = copy.deepcopy(describe) 
            ind = ind + 1 
        path =""
        if cmd._config_dict['method'] == 'train':
            path = cmd._config_dict['train_dir']
        else:
            path = cmd._config_dict['eval_dir']
        with open(os.path.join(path,"describe.json"),"w") as outfile:
            json.dump(self.task_dict,outfile)
        get_ckpt_to_keep(path)

class Config_Writer:
    def __init__(self,folder):
        self.folder = folder 
    def write(self,file_name,config_dict):
        path = os.path.join(self.folder,file_name)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        with open(os.path.join(self.folder,file_name),"w") as f:
            json.dump(config_dict,f)

class ConfigFactory:
    def __init__(self):
        pass
    def _gen_mobile_net_train(self,checkpoint_path, dataset_dir, train_dir,dataset_name="rcmp",batch_size=32,dataset_split_name="train"):
        train_dict = {}
        train_dict["method"]="train"
        train_dict["train_dir"] = train_dir
        train_dict["dataset_dir"] = dataset_dir
        train_dict["dataset_name"]=dataset_name
        train_dict["dataset_split_name"]=dataset_split_name
        train_dict["model_name"]="mobilenet_v1"
        train_dict["checkpoint_path"]=checkpoint_path
        train_dict["max_number_of_steps"]="50"
        train_dict["batch_size"]=str(batch_size)
        train_dict["learning_rate"]="0.0001"
        train_dict["learning_rate_decay_type"]="fixed"
        train_dict["optimizer"]="rmsprop"
        train_dict["weight_decay"]="0.00004"
        train_dict["preprocessing_name"]="inception"
        train_dict["labels_offset"]="0"
        #the following two scopes need spcial attention. 
        train_dict["checkpoint_exclude_scopes"]="MobilenetV1/Logits,MobilenetV1/AuxLogits"
        train_dict["trainable_scopes"]="MobilenetV1/Logits,MobilenetV1/AuxLogits"
        train_dict["save_interval_secs"]="600" 
        train_dict["save_summaries_secs"]="60" 
        return train_dict 
    def _gen_common_train(self,checkpoint_path, dataset_dir, train_dir,dataset_name="rcmp",batch_size=32,dataset_split_name="train"):
        train_dict={}
        train_dict["method"]="train"
        train_dict["train_dir"] = train_dir
        train_dict["dataset_dir"] = dataset_dir
        train_dict["dataset_name"]=dataset_name
        train_dict["dataset_split_name"]=dataset_split_name
        train_dict["batch_size"]=str(batch_size)

        train_dict["optimizer"]="rmsprop"
        train_dict["weight_decay"]="0.00004"
        train_dict["learning_rate_decay_type"]="fixed"
        train_dict["labels_offset"]="0"
        
        train_dict["preprocessing_name"]="inception"
        
        train_dict["save_interval_secs"]="600" 
        train_dict["save_summaries_secs"]="60" 
        
        return train_dict
    def _gen_common_eval(self,checkpoint_path,dataset_dir,eval_dir,dataset_name="rcmp",dataset_split_name="validation"):
        """
        usually make eval_dir == checkpoint_dir for evaluation 
        """
        example_dict={}
        example_dict["method"] = "eval"
        example_dict["eval_dir"] = eval_dir
        example_dict["dataset_name"] = dataset_name
        example_dict["dataset_dir"] = dataset_dir
        example_dict["dataset_split_name"] = dataset_split_name
        example_dict["checkpoint_path"] = checkpoint_path
        example_dict["labels_offset"]="0"
        return example_dict
    
    def _gen_inception_v3_train(self,checkpoint_path, dataset_dir, train_dir,dataset_name="rcmp",batch_size=32,dataset_split_name="train"):
        train_dict = self._gen_common_train(checkpoint_path, dataset_dir, train_dir,dataset_name,batch_size,dataset_split_name)
        train_dict["model_name"]="inception_v3"
        train_dict["learning_rate"] = "0.01"
        train_dict["max_number_of_steps"] = "50"
        train_dict["trainable_scopes"] ="InceptionV3/Logits,InceptionV3/AuxLogits"
        train_dict["checkpoint_exclude_scopes"] ="InceptionV3/Logits,InceptionV3/AuxLogits"
        return train_dict
    
    def _gen_inception_v3_eval(self,checkpoint_path,dataset_dir,eval_dir,dataset_name="rcmp",dataset_split_name="validation"):
        example_dict  = self._gen_common_eval(checkpoint_path, dataset_dir, eval_dir,dataset_name,dataset_split_name)
        example_dict["model_name"]="inception_v3"
        return example_dict
     
    def _gen_inception_resnet_v2_train(self,checkpoint_path, dataset_dir, train_dir,dataset_name="rcmp",batch_size=32,dataset_split_name="train"):
        train_dict = self._gen_common_train(checkpoint_path, dataset_dir, train_dir,dataset_name,batch_size,dataset_split_name)
        train_dict["model_name"]="inception_resnet_v2"
        train_dict["learning_rate"] = "0.01"
        train_dict["max_number_of_steps"] = "50"
        train_dict["trainable_scopes"] ="InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits"
        train_dict["checkpoint_exclude_scopes"] ="InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits"
        return train_dict
    
    def _gen_inception_resnet_v2_eval(self,checkpoint_path,dataset_dir,eval_dir,dataset_name="rcmp",dataset_split_name="validation"):
        example_dict  = self._gen_common_eval(checkpoint_path, dataset_dir, eval_dir,dataset_name,dataset_split_name)
        example_dict["model_name"]="inception_resnet_v2"
        return example_dict
    
    def _gen_mobile_net_eval(self,checkpoint_path,dataset_dir,eval_dir,dataset_name="rcmp",dataset_split_name="validation"):
        """
        usually make eval_dir == checkpoint_dir for evaluation 
        """
        example_dict={}
        example_dict["method"] = "eval"
        example_dict["eval_dir"] = eval_dir
        example_dict["dataset_name"] = dataset_name
        example_dict["model_name"] ="mobilenet_v1"
        example_dict["dataset_dir"] = dataset_dir
        example_dict["dataset_split_name"] = dataset_split_name
        example_dict["checkpoint_path"] = checkpoint_path
        example_dict["labels_offset"]="0"
        return example_dict
    def _gen_default(self):
        de = {}
        return de 
    def gen_fn(self,task_type):  
        factory_map = {
            "eval:mobile_net_v1": self._gen_mobile_net_eval,
            "train:mobile_net_v1": self._gen_mobile_net_train,
            "eval:inception_resnet_v2":self._gen_inception_resnet_v2_eval,
            "train:inception_resnet_v2":self._gen_inception_resnet_v2_train,
            "eval:inception_v3":self._gen_inception_v3_eval,
            "train:inception_v3":self._gen_inception_v3_train,
            
        }
        if task_type in factory_map.keys():
            return factory_map[task_type]
        else:
            return self._gen_default

def path_gen(ckpt,train_folder,csv_name_prefix):
    train_path = os.path.join(".", train_folder)
    csv_path = os.path.join(train_path,csv_name_prefix)
    return ckpt, train_path, train_path,csv_path

import re
import os, sys

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def get_ckpt_to_keep(train_dir,max_to_keep=3):
    all_parts = splitall(train_dir)
    pre = all_parts[0:(len(all_parts)-1)]
    dir_name = all_parts[len(all_parts)-1]
    dir_name = "useless_" + dir_name
    pre.append(dir_name)
    path = os.path.join(*pre)
    print(path)
    with open(os.path.join(train_dir,"checkpoint")) as ckfile:
        content = ckfile.readlines()
    l = [x.strip() for x in content]
    pattern = re.compile(r"model\.ckpt-\d+")
    files = map(lambda x : pattern.search(x).group(), l)
    files = files[1:len(files)]
    remove_files = []
    
    if len(files) <=max_to_keep:
        pass
    else:
        remove_files = files[0:(len(files)-max_to_keep)]
    print(remove_files)
    os.system("mkdir "+ path)
    for file in remove_files:
        file_path = os.path.join(train_dir,file)
        cmd = "mv "+ file_path + "* " + path
        print(cmd)
        os.system("mv "+ file_path + "* " + path)

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
    train_config_frozen["max_number_of_steps"] = 31250
    train_config_unfrozen = copy.deepcopy(train_config)
    train_config_unfrozen['checkpoint_path'] = train_config_frozen['train_dir']
    train_config_unfrozen["learning_rate"]="0.0001"
    train_config_unfrozen["max_number_of_steps"] = 218750
    return train_config_frozen, train_config_unfrozen