import os.path

import torch
import json
import torch.backends.cudnn as cudnn

from config import args
from evaluate import predict_by_split
from trainer import Trainer
from logger_config import logger

import wandb
import time

### hmt
from dict_hub import get_train_triplet_dict
import os
import shutil
### end hmt


def copy_py_code(orignal_path, target_path):
    file_names = os.listdir(orignal_path)
    print(file_names)
    for fname in file_names:
        if fname.endswith('.py'):
            shutil.copyfile(os.path.join(orignal_path, fname),
                            os.path.join(target_path, fname))
    print(f'复制文件完毕，已复制到 {target_path}')

def main():
    wandb_logger = None
    if args.__dict__['use_wandb']:
        args.__dict__['addName'] = args.__dict__['wandb_name'] + '_' +time.strftime('%Y%m%d_%H%M',
                                                                               time.localtime(time.time()))
        if not args.__dict__['sweep']:
            args.__dict__['model_dir'] = os.path.join(args.__dict__['model_dir'], args.__dict__['addName'])
            if args.model_dir:
                os.makedirs(args.model_dir, exist_ok=True)
            else:
                assert os.path.exists(
                    args.eval_model_path), 'One of args.model_dir and args.eval_model_path should be valid path'
                args.model_dir = os.path.dirname(args.eval_model_path)
        wandb_logger = wandb.init(entity='homurat', project=args.__dict__['project_name'],
                                  name=args.__dict__['addName'])

        copy_py_code('./', args.model_dir)


        for k, v in args.__dict__.items():
            wandb_logger.config[k] = v

    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    ### hmt set relation number
    args.__dict__['rel_num'] = get_train_triplet_dict().relations.__len__()
    ### end hmt
    args.is_test = False
    trainer = Trainer(args, ngpus_per_node=ngpus_per_node, wandb_logger=wandb_logger)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()

    ###hmt

    args.is_test = True
    print('set test path')
    args.valid_path = args.test_path
    args.eval_model_path = os.path.join(args.__dict__['model_dir'], 'model_last.mdl')

    # save args
    print(f"保存训练配置:{os.path.join(args.__dict__['model_dir'], 'train_args.bin')}")
    torch.save(args, os.path.join(args.__dict__['model_dir'], 'train_args.bin'))

    predict_by_split(wandb_logger=wandb_logger)

    ### end hmt
    if wandb_logger:
        wandb_logger.finish()


if __name__ == '__main__':
    main()
