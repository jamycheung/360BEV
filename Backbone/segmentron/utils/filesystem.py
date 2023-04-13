"""Filesystem utility functions."""
from __future__ import absolute_import
import os, glob
import errno
import torch
import logging

from ..config import cfg

def save_checkpoint(args, model, epoch, optimizer=None, lr_scheduler=None, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg.TRAIN.MODEL_SAVE_DIR)
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    filename = '{}_epoch_{}.pth'.format(cfg.TIME_STAMP, str(epoch))
    if is_best:
        best_filename = 'best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        torch.save(model_state_dict, best_filename)
    else:
        pre_filename = glob.glob('{}*.pth'.format(cfg.TIME_STAMP))
        try:
            for p in pre_filename:
                os.remove(p)
        except OSError as e:
            logging.info(e)

        # save epoch
        save_state = {
            'epoch': epoch,
            'state_dict': model_state_dict,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
        filename = os.path.join(directory, filename)
        if not args.distributed or (args.distributed and args.local_rank % args.num_gpus == 0):
            torch.save(save_state, filename)
            # logging.info('Epoch {} model saved in: {}'.format(epoch, filename))

def makedirs(path):
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.
    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

