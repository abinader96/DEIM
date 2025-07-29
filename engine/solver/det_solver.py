"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):
    
    def _log_validation_metrics(self, test_stats, epoch):
        """Log validation metrics to TensorBoard."""
        if not (self.writer and dist_utils.is_main_process()):
            return
            
        for k, v in test_stats.items():
            if k.startswith('loss'):
                # Log scalar loss values
                self.writer.add_scalar(f'Val/{k}', v, epoch)
            elif k == 'coco_eval_bbox':
                # Log COCO evaluation metrics
                metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 
                              'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
                for metric_name, metric_value in zip(metric_names, v):
                    self.writer.add_scalar(f'Val/COCO_{metric_name}', metric_value, epoch)
    
    def _save_best_model(self, epoch, stage='stg1'):
        """Save the best model checkpoint."""
        if self.output_dir:
            filename = f'best_{stage}.pth'
            dist_utils.save_on_master(self.state_dict(), self.output_dir / filename)
    
    def _update_best_stats(self, test_stats, best_stat, best_stat_print, top1, epoch):
        """Update best statistics and save models if needed."""
        for k in test_stats:
            if k != 'coco_eval_bbox':
                continue
                
            # Use AP (first value) as the main metric
            ap_value = test_stats[k][0]
            
            # Update best_stat
            if k in best_stat:
                if ap_value > best_stat[k]:
                    best_stat['epoch'] = epoch
                    best_stat[k] = ap_value
            else:
                best_stat['epoch'] = epoch
                best_stat[k] = ap_value
            
            # Check if this is the new best
            if best_stat[k] > top1:
                best_stat_print['epoch'] = epoch
                top1 = best_stat[k]
                stage = 'stg2' if epoch >= self.train_dataloader.collate_fn.stop_epoch else 'stg1'
                self._save_best_model(epoch, stage)
            
            best_stat_print[k] = max(best_stat.get(k, 0), top1)
            print(f'best_stat: {best_stat_print}')
            
            # Additional save logic for current epoch
            if best_stat['epoch'] == epoch and self.output_dir:
                if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if ap_value > top1:
                        top1 = ap_value
                        self._save_best_model(epoch, 'stg2')
                else:
                    top1 = max(ap_value, top1)
                    self._save_best_model(epoch, 'stg1')
            
            # Reset logic for stage 2
            elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                best_stat = {'epoch': -1}
                self.ema.decay -= 0.0001
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')
                
        return best_stat, best_stat_print, top1
    
    def _save_checkpoints(self, epoch):
        """Save regular checkpoints during training."""
        if not self.output_dir or epoch >= self.train_dataloader.collate_fn.stop_epoch:
            return
            
        checkpoint_paths = [self.output_dir / 'last.pth']
        if (epoch + 1) % self.cfg.checkpoint_freq == 0:
            checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
            
        for checkpoint_path in checkpoint_paths:
            dist_utils.save_on_master(self.state_dict(), checkpoint_path)
    
    def _log_training_stats(self, train_stats, test_stats, epoch, n_parameters):
        """Log training statistics to file."""
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        
        if self.output_dir and dist_utils.is_main_process():
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    def _save_eval_logs(self, coco_evaluator, epoch):
        """Save evaluation logs."""
        if not (self.output_dir and dist_utils.is_main_process() and coco_evaluator):
            return
            
        (self.output_dir / 'eval').mkdir(exist_ok=True)
        if "bbox" in coco_evaluator.coco_eval:
            filenames = ['latest.pth']
            if epoch % 50 == 0:
                filenames.append(f'{epoch:03}.pth')
            for name in filenames:
                torch.save(coco_evaluator.coco_eval["bbox"].eval,
                          self.output_dir / "eval" / name)

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches, 
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            best_stat['epoch'] = self.last_epoch
            top1 = best_stat["coco_eval_bbox"] = test_stats["coco_eval_bbox"][0]
            print(f'best_stat: {best_stat}')         

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            
            # Save regular checkpoints
            self._save_checkpoints(epoch)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # Log validation metrics to TensorBoard
            self._log_validation_metrics(test_stats, epoch)
            
            # Update best statistics and save models
            best_stat, best_stat_print, top1 = self._update_best_stats(
                test_stats, best_stat, best_stat_print, top1, epoch
            )
            
            # Log training statistics
            self._log_training_stats(train_stats, test_stats, epoch, n_parameters)
            
            # Save evaluation logs
            self._save_eval_logs(coco_evaluator, epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
