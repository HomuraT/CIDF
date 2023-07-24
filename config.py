import os
import random
import torch
import argparse
import warnings

import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='SimKGC arguments')
parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--task', default='wn18rr', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--train-path', default='', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--model-dir', default='', type=str, metavar='N',
                    help='path to model dir')
parser.add_argument('--warmup', default=400, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--max-to-keep', default=5, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--pooling', default='cls', type=str, metavar='N',
                    help='bert pooling')
parser.add_argument('--dropout', default=0.1, type=float, metavar='N',
                    help='dropout on final linear layer')
parser.add_argument('--use-amp', action='store_true',
                    help='Use amp if available')
parser.add_argument('--t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--use-link-graph', action='store_true',
                    help='use neighbors from link graph as context')
parser.add_argument('--eval-every-n-step', default=10000, type=int,
                    help='evaluate every n steps')
parser.add_argument('--pre-batch', default=0, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                    help='the weight for logits from pre-batch negatives')
parser.add_argument('--additive-margin', default=0.0, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--finetune-t', action='store_true',
                    help='make temperature as a trainable parameter or not')
parser.add_argument('--max-num-tokens', default=50, type=int,
                    help='maximum number of tokens')
parser.add_argument('--use-self-negative', action='store_true',
                    help='use head entity as negative')

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# only used for evaluation
parser.add_argument('--is-test', action='store_true',
                    help='is in test mode or not')
parser.add_argument('--rerank-n-hop', default=2, type=int,
                    help='use n-hops node for re-ranking entities, only used during evaluation')
parser.add_argument('--neighbor-weight', default=0.0, type=float,
                    help='weight for re-ranking entities')
parser.add_argument('--eval-model-path', default='', type=str, metavar='N',
                    help='path to model, only used for evaluation')

### hmt
parser.add_argument('-swp', '--sweep', help='sweep config name', required=False, default=None, type=str)
parser.add_argument('-swpc', '--sweep_count', help='sweep count', required=False, default=0, type=int)
parser.add_argument('-sweep_name', '--sweep_name', help='sweep_name', required=False)
parser.add_argument('-wname', '--wandb_name', help='wanbd_name', required=False, default=None, type=str)
parser.add_argument('-uw', '--use_wandb', action='store_true', help='使用wandb')
parser.add_argument('--project_name', help='wandb project name', required=False, default='KGC')
parser.add_argument('--test-path', default='', type=str, metavar='N',
                    help='path to test data')

parser.add_argument('-pl', '--prefix_length', help='length of predix length', required=False, default=10, type=int)
parser.add_argument('-sclw', '--self_cl_weight', help='weight of scl', required=False, default=0, type=float)
parser.add_argument('-vl2w', '--vector_regular_weight', help='vector_regular_weight', required=False, default=0, type=float)
parser.add_argument('-drpath', '--detailed_relation_description', help='detailed_relation_description', required=False, type=str)

parser.add_argument('-sm', '--score_mode', help='计算得分方式 cos-sim minus-distance', required=False, type=str, default='cos-sim')

parser.add_argument('-apath', '--args_path', help='配置文件的路径', required=False, type=str)
parser.add_argument('-mpath', '--model_path', help='模型路径', required=False, type=str)


args = parser.parse_args()

if args.args_path:
    print(f'载入存储的训练配置文件：{args.args_path}')
    training_args = torch.load(args.args_path)
    for k, v in training_args.__dict__.items():
        if k != 'is_test':
            args.__dict__[k] = v
    args.use_wandb = False
    args.valid_path = args.test_path
    if args.model_path:
        args.model_dir= os.path.dirname(args.model_path)
        args.eval_model_path=args.model_path


assert not args.train_path or os.path.exists(args.train_path)
assert args.pooling in ['cls', 'mean', 'max']
assert args.task.lower() in ['wn18rr', 'fb15k237', 'wiki5m_ind', 'wiki5m_trans']
assert args.lr_scheduler in ['linear', 'cosine']

if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True)
else:
    assert os.path.exists(args.eval_model_path), 'One of args.model_dir and args.eval_model_path should be valid path'
    args.model_dir = os.path.dirname(args.eval_model_path)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn('GPU is not available, set use_amp=False and print_freq=1')
