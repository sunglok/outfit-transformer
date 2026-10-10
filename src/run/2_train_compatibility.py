import json
import logging
import os
import pathlib
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb

from ..data import collate_fn
from ..data.datasets import polyvore
from ..evaluation.metrics import compute_cp_scores
from ..models.load import load_model
from ..utils.distributed_utils import cleanup, gather_results, setup
from ..utils.logger import get_logger
from ..utils.loss import FocalLoss
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
CHECKPOINT_DIR = SRC_DIR / 'checkpoints'
RESULT_DIR = SRC_DIR / 'results'
LOGS_DIR = SRC_DIR / 'logs'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str,
                        choices=['original', 'clip', 'resnet18', 'resnet18_finetune'],
                        default='clip')
    parser.add_argument('--polyvore_dir', type=str, 
                        default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'],
                        default='nondisjoint')
    parser.add_argument('--batch_sz_per_gpu', type=int,
                        default=512)
    parser.add_argument('--n_workers_per_gpu', type=int,
                        default=4)
    parser.add_argument('--n_epochs', type=int,
                        default=200)
    parser.add_argument('--lr', type=float,
                        default=2e-5)
    parser.add_argument('--accumulation_steps', type=int,
                        default=4)
    parser.add_argument('--wandb_key', type=str, 
                        default=None)
    parser.add_argument('--seed', type=int, 
                        default=42)
    parser.add_argument('--checkpoint', type=str, 
                        default=None)
    parser.add_argument('--resume', type=str,
                        default=None,
                        help='Resume model, optimizer, scheduler, epoch, and early-stopping state from a training checkpoint.')
    parser.add_argument('--world_size', type=int, 
                        default=-1)
    parser.add_argument('--project_name', type=str, 
                        default=None)
    parser.add_argument('--early_stopping_patience', type=int,
                        default=0,
                        help='Stop after this many epochs without validation AUC improvement; 0 disables it.')
    parser.add_argument('--demo', action='store_true')
    
    return parser.parse_args()


def setup_dataloaders(rank, world_size, args):
    metadata = polyvore.load_metadata(args.polyvore_dir)
    use_precomputed_embedding = args.model_type in {'clip', 'resnet18'}
    embedding_dict = (
        polyvore.load_embedding_dict(args.polyvore_dir)
        if use_precomputed_embedding else None
    )
    
    train = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type, 
        dataset_split='train', metadata=metadata,
        load_image=not use_precomputed_embedding, embedding_dict=embedding_dict
    )
    valid = polyvore.PolyvoreCompatibilityDataset(
        dataset_dir=args.polyvore_dir, dataset_type=args.polyvore_type, 
        dataset_split='valid', metadata=metadata,
        load_image=not use_precomputed_embedding, embedding_dict=embedding_dict
    )
    
    if world_size == 1:
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=True,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn,
            persistent_workers=args.n_workers_per_gpu > 0
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn,
            persistent_workers=args.n_workers_per_gpu > 0
        )
        
    else:
        train_sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        valid_sampler = DistributedSampler(
            valid, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_dataloader = DataLoader(
            dataset=train, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn, sampler=train_sampler,
            persistent_workers=args.n_workers_per_gpu > 0
        )
        valid_dataloader = DataLoader(
            dataset=valid, batch_size=args.batch_sz_per_gpu, shuffle=False,
            num_workers=args.n_workers_per_gpu, collate_fn=collate_fn.cp_collate_fn, sampler=valid_sampler,
            persistent_workers=args.n_workers_per_gpu > 0
        )

    return train_dataloader, valid_dataloader


def train_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, optimizer, scheduler, loss_fn, dataloader
):
    model.train()  
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{args.n_epochs}', disable=(rank != 0))
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        queries = data['query']
        labels = torch.tensor(data['label'], dtype=torch.float32).to(rank)
        
        preds = model(
            queries,
            use_precomputed_embedding=args.model_type in {'clip', 'resnet18'},
        ).squeeze(1)
        
        loss = loss_fn(y_true=labels, y_prob=preds) / args.accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging 
        score = compute_cp_scores(all_preds[-1], all_labels[-1])
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            'lr': scheduler.get_last_lr()[0] if scheduler else args.lr,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'train_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
    

    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {'loss': gathered_loss.item(), **compute_cp_scores(gathered_preds, gathered_labels)} if rank == 0 else {}
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return {f'train_{key}': value for key, value in output.items()}
   
        
@torch.no_grad()
def valid_step(
    rank, world_size, 
    args, epoch, logger, wandb_run,
    model, loss_fn, dataloader
):
    model.eval()
    pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch+1}/{args.n_epochs}', disable=(rank != 0))
    
    all_loss, all_preds, all_labels = torch.zeros(1, device=rank), [], []
    for i, data in enumerate(pbar):
        if args.demo and i > 2:
            break
        queries = data['query']
        labels = torch.tensor(data['label'], dtype=torch.float32).to(rank)
    
        preds = model(
            queries,
            use_precomputed_embedding=args.model_type in {'clip', 'resnet18'},
        ).squeeze(1)
        
        loss = loss_fn(y_true=labels, y_prob=preds) / args.accumulation_steps
        
        # Accumulate Results
        all_loss += loss.item() * args.accumulation_steps / len(dataloader)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

        # Logging
        score = compute_cp_scores(all_preds[-1], all_labels[-1])
        logs = {
            'loss': loss.item() * args.accumulation_steps,
            'steps': len(pbar) * epoch + i,
            **score
        }
        pbar.set_postfix(**logs)
        if args.wandb_key and rank == 0:
            logs = {f'valid_{k}': v for k, v in logs.items()}
            wandb_run.log(logs)
        
    
    all_preds = torch.cat(all_preds).to(rank)
    all_labels = torch.cat(all_labels).to(rank)

    gathered_loss, gathered_preds, gathered_labels = gather_results(all_loss, all_preds, all_labels)
    output = {}
    if rank == 0:
        all_score = compute_cp_scores(gathered_preds, gathered_labels)
        output = {'loss': gathered_loss.item(), **all_score}
        
    logger.info(f'Epoch {epoch+1}/{args.n_epochs} --> End {output}')

    return {f'valid_{key}': value for key, value in output.items()}


def train(
    rank: int, world_size: int, args: Any,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
):  
    # Setup
    setup(rank, world_size)
    
    # Logging Setup
    project_name = f'compatibility_{args.model_type}_' + (
        args.project_name if args.project_name 
        else (wandb_run.name if wandb_run else 'test')
    )
    logger = get_logger(project_name, LOGS_DIR, rank)
    logger.info(f'Logger Setup Completed')
    
    # Dataloaders
    train_dataloader, valid_dataloader = setup_dataloaders(rank, world_size, args)
    logger.info(f'Dataloaders Setup Completed')
    
    # Model setting
    model = load_model(model_type=args.model_type, checkpoint=args.resume or args.checkpoint)
    logger.info(f'Model Loaded and Wrapped with DDP')
    
    # Optimizer, Scheduler, Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr, epochs=args.n_epochs, steps_per_epoch=int(len(train_dataloader) / args.accumulation_steps),
        pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4
    )
    loss_fn = FocalLoss(alpha=0.5, gamma=2) # focal_loss(alpha=0.5, gamma=2)
    logger.info(f'Optimizer and Scheduler Setup Completed')

    # Training Loop
    start_epoch = 0
    best_valid_auc = float('-inf')
    epochs_without_improvement = 0
    if args.resume:
        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        resume_state = torch.load(args.resume, map_location=map_location, weights_only=False)
        required_keys = {'optimizer', 'scheduler', 'epoch'}
        missing_keys = required_keys.difference(resume_state)
        if missing_keys:
            raise ValueError(
                f'Checkpoint {args.resume} cannot resume training; missing keys: '
                f'{sorted(missing_keys)}. Use --checkpoint for weights-only initialization.'
            )
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        start_epoch = int(resume_state['epoch'])
        best_valid_auc = float(resume_state.get('best_valid_auc', float('-inf')))
        epochs_without_improvement = int(resume_state.get('epochs_without_improvement', 0))
        logger.info(f'Resuming training at epoch {start_epoch + 1}/{args.n_epochs}')

    for epoch in range(start_epoch, args.n_epochs):
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)
        train_logs = train_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            model, optimizer, scheduler, loss_fn, train_dataloader
        )
        
        valid_logs = valid_step(
            rank, world_size, 
            args, epoch, logger, wandb_run,
            model, loss_fn, valid_dataloader
        )
        
        checkpoint_dir = CHECKPOINT_DIR / project_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            
        if rank == 0:
            torch.save({
                'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_valid_auc': max(best_valid_auc, valid_logs['valid_auc']),
                'epochs_without_improvement': (
                    0 if valid_logs['valid_auc'] > best_valid_auc
                    else epochs_without_improvement + 1
                ),
            }, checkpoint_path)
            
            score_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_score.json')
            with open(score_path, 'w') as f:
                score = {**train_logs, **valid_logs}
                json.dump(score, f, indent=4)
            logger.info(f'Checkpoint saved at {checkpoint_path}')

            valid_auc = valid_logs['valid_auc']
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                epochs_without_improvement = 0
                best_path = os.path.join(checkpoint_dir, 'best.pth')
                torch.save({
                    'config': model.module.cfg.__dict__ if world_size > 1 else model.cfg.__dict__,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'valid_auc': valid_auc,
                    'best_valid_auc': valid_auc,
                    'epochs_without_improvement': 0,
                }, best_path)
                logger.info(f'Best checkpoint updated at {best_path} (AUC={valid_auc:.6f})')
            else:
                epochs_without_improvement += 1
            
        dist.barrier()
        map_location = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        model.load_state_dict(state_dict['model'])
        logger.info(f'Checkpoint loaded from {checkpoint_path}')

        should_stop = rank == 0 and args.early_stopping_patience > 0 and (
            epochs_without_improvement >= args.early_stopping_patience
        )
        stop_tensor = torch.tensor(int(should_stop), device=rank)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            logger.info(
                f'Early stopping after {epochs_without_improvement} epochs without '
                f'validation AUC improvement (best={best_valid_auc:.6f})'
            )
            break

        if args.demo:
            break
        
    cleanup()


if __name__ == '__main__':
    args = parse_args()
    
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(project='outfit-transformer-cp', config=args.__dict__)
    else:
        wandb_run = None
        
    if args.world_size == 1:
        train(0, 1, args, wandb_run)
    else:
        mp.spawn(train, args=(args.world_size, args, wandb_run), nprocs=args.world_size, join=True)
