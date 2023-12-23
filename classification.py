#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import wandb
import argparse

import torch
import numpy as np
from defaults import *
from utils.system_def import *
from utils.launch import dist, launch, synchronize
from utils.helpfuns import check_dir

global debug


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as \
                             argument the parameters dictionary from a json file')
    parser.add_argument('--params_path', type=str, required=False, 
                        default="./params.json",
                        help='Give the path of the json file which contains the training parameters')
    parser.add_argument('--checkpoint', type=str, required=False, help='Give a valid checkpoint name')
    parser.add_argument('--test', action='store_true', default=False, help='Flag for testing')
    parser.add_argument('--lr', type=float, help='Change learning rate')
    parser.add_argument('--debug', action='store_true', default=False, help='Flag for turning on the debug_mode')
    parser.add_argument('--data_location', type=str, required=False, help='Update the datapath')
    parser.add_argument('--dist_url', type=str, default='', required=False,
                        help='URL of master node, for use with SLURM')
    parser.add_argument('--port', type=int, required=False, default=45124, 
                        help='Explicit port selection, for use with SLURM')
    parser.add_argument('--gpu', type=str, required=False, help='The GPU to be used for this run')
    parser.add_argument('--model_name', type=str, required=False, help='Used to manipulate the model_name defined in the param file for this run')
    parser.add_argument('--save_dir', type=str, required=False, help='Change the "save_dir" in param file')
    parser.add_argument('--backbone_type', type=str, help='Change backbone')
    parser.add_argument('--batch_size', type=int, help='Change the batch size of all data loaders')
    parser.add_argument('--patch_size', type=int, help='Change model patch size')
    parser.add_argument('--val_every', type=float, help='How many epochs between each validation')
    parser.add_argument('--use_mixed_precision', type=int, default=-1, help='Using this arg with value 1 (0) results in explicity turning on (off) the mixed precision. By default it follows the param file.')
   
    return parser.parse_args()


def update_params_from_args(params, args):
    if args.gpu:
        prev_gpu = params.system_params.which_GPUs
        params.system_params.which_GPUs = args.gpu  # change the value in-place
        print('Changed GPU for this run from {} to \033[1m{}\033[0m'.format(prev_gpu, args.gpu))

    if args.lr:
        prev_lr = params.optimization_params.default.optimizer.params.lr
        params.optimization_params.default.optimizer.params.lr = args.lr  # change the value in-place
        print('Changed GPU for this run from {} to \033[1m{}\033[0m'.format(prev_lr, args.lr))

    if args.model_name:
        prev_model_name = params.training_params.model_name
        params.training_params.model_name = args.model_name
        print('Changed model_name for this run from {} to \033[1m{}\033[0m'.format(prev_model_name, args.model_name))

    if args.data_location:
        params['dataset_params']['data_location'] = args.data_location
        print('Changed data_location to: "\033[1m{}\033[0m"'.format(args.data_location))

    if args.save_dir:
        params['training_params']['save_dir'] = args.save_dir
        print('Changed save_dir to: "\033[1m{}\033[0m"'.format(args.save_dir))

    if args.backbone_type:
        params['model_params']['backbone_type'] = args.backbone_type
        print('Changed backbone_type to: \033[1m{}\033[0m'.format(args.backbone_type))

    if args.batch_size:
        for loader in ['trainloader', 'valloader', 'testloader']:
            params['dataloader_params'][loader]['batch_size'] = args.batch_size
            print('Changed \033[1m{}\033[0m batch_size to: \033[1m{}\033[0m'.format(loader, args.batch_size))

    if args.patch_size:
        params['model_params']['transformers_params']['patch_size'] = args.patch_size
        print('Changed patch_size to: \033[1m{}\033[0m'.format(args.patch_size))

    if args.val_every is not None:
        params['training_params']['val_every'] = args.val_every
        print('Changed val_every to: \033[1m{}\033[0m'.format(args.val_every))

    if args.use_mixed_precision != -1:  # if -1, it won't change the param file
        assert args.use_mixed_precision == 0 or args.use_mixed_precision == 1, 'Argument --use_mixed_precision shold be either 1 or 0'
        if_mixed = bool(args.use_mixed_precision)  # this step is not crucial but is good for consistency
        params['training_params']['use_mixed_precision'] = if_mixed
        print('Explicitly set use_mixed_precision to: \033[1m{}\033[0m'.format(if_mixed))

    if args.lr:
        params['optimization_params']['default']['optimizer']['params']['lr'] = args.lr
        print('Changed learning rate to: \033[1m{}\033[0m'.format(args.lr))

def save_config(params_file, save_dir, model_name):
    path = os.path.join(save_dir, "configs", model_name) 
    check_dir(path)
    shutil.copy(params_file,path)

def update_warmup(params):
    if params["dataset_params"]["dataset"] == 'DDSM' or params["dataset_params"]["dataset"] == 'APTOS2019':
        params["optimization_params"]["default"]["scheduler"]["params"]["LinearWarmup"]["warmup_epochs"] = 5
    else:
        params["optimization_params"]["default"]["scheduler"]["params"]["LinearWarmup"]["warmup_epochs"] = 0

    if params["dataset_params"]["dataset"] == 'ISIC2019' or params["dataset_params"]["dataset"] == 'CheXpert':
        params["optimization_params"]["default"]["scheduler"]["params"]["LinearWarmup"]["warmup_iters"] = 1000
    else:
        params["optimization_params"]["default"]["scheduler"]["params"]["LinearWarmup"]["warmup_iters"] = 0


def main(parameters, args):
    
    # define system
    define_system_params(parameters.system_params)

    # Instantiate wrapper with all its definitions 
    if parameters.model_params.backbone_type == "none" :
        print("Using foundation wrapper")
        wrapper = FoundationWrapper(parameters)
        wrapper.instantiate()
    else:
        if parameters.foundation_params.backbone_type == "none":
            wrapper = DefaultWrapper(parameters)
        elif "focal" in parameters.foundation_params.backbone_type:
            wrapper = SEEMWrapper(parameters)
        elif "sam" in parameters.foundation_params.backbone_type:
            wrapper = SAMWrapper(parameters)
        elif "dinov2" in parameters.foundation_params.backbone_type:
            wrapper = DINOWrapper(parameters)
        elif "clip" in parameters.foundation_params.backbone_type:
            wrapper = CLIPWrapper(parameters)
        elif "blip" in parameters.foundation_params.backbone_type:
            wrapper = BLIPWrapper(parameters)
        else:
            raise NotImplementedError(f"{parameters.foundation_params.backbone_type} not implemented")
        print(f"Using {parameters.foundation_params.backbone_type} wrapper")
        wrapper.instantiate()
    
    # initialize logger
    if wrapper.is_rank0:
        log_params = wrapper.parameters.log_params    
        training_params = wrapper.parameters.training_params
        if wrapper.log_params['run_name'] == "DEFINED_BY_MODEL_NAME":
            log_params['run_name'] = training_params.model_name  
        if args.debug:
            os.environ['WANDB_MODE'] = 'dryrun'
        if not args.test:
            if parameters.training_params.use_tensorboard:
                print("Using TensorBoard logging")
                summary_writer = SummaryWriter()
            else:
                print("Using WANDB logging")
                wandb.init(project=log_params.project_name, 
                           entity='your_entity',
                           name=log_params.run_name, 
                           config=wrapper.parameters,
                           resume=True if training_params.restore_session else False)
    
    # define trainer 
    trainer = Trainer(wrapper)
        
    if parameters.training_params.use_tensorboard:
        trainer.summary_writer = summary_writer
        
    if args.test:
        trainer.test()     
    else:
        trainer.train()
        if wrapper.is_supervised:
            trainer.test()
        
    
if __name__ == '__main__':
    args = parse_arguments()
    parameters = edict(load_params(args))
    update_params_from_args(parameters, args)
    update_warmup(parameters)

    try:
        launch(main, (parameters, args))
    except Exception as e:       
        if dist.is_initialized():
            dist.destroy_process_group()            
        raise e
    finally:
        if dist.is_initialized():
            synchronize()         
            dist.destroy_process_group()            
