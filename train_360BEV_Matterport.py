import datetime
import os
import yaml
import time
import torch
import shutil
import random
import argparse
import numpy as np

import torch.multiprocessing as mp
from pathlib import Path
from torch.utils import data
import torch.distributed as distrib
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler
from model.dataloader_mp3d.pano_data_loader import DatasetLoader_pano_detr

# from model.trans4pano_map_new_decoder import Trans4map_segformer
from model.BEV360_segformer_matterport import BEV360_segformer
from model.BEV360_segnext_matterport import BEV360_segnext

from model.loss import SemmapLoss
from metric import averageMeter
from metric.iou import IoU
from model.other_models.utils import get_logger

def train(rank, world_size, cfg):
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # init distributed compute
    master_port = int(os.environ.get("MASTER_PORT", 8738))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    tcp_store = torch.distributed.TCPStore(
        master_addr, master_port, world_size, rank == 0
    )
    torch.distributed.init_process_group(
        'nccl', store=tcp_store, rank=rank, world_size=world_size
    )

    ################################################## Setup device ####################################################
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        assert world_size == 1
        device = torch.device("cpu")

    if rank == 0:
        writer = SummaryWriter(logdir=cfg["logdir"])
        logger = get_logger(cfg["logdir"])

        print('**log_dir:', cfg["logdir"])
        logger.info("Let Trans4Map training begin !!")

    t_loader = DatasetLoader_pano_detr(cfg["data"], split=cfg['data']['train_split'])
    v_loader = DatasetLoader_pano_detr(cfg['data'], split=cfg["data"]["val_split"])

    t_sampler = DistributedSampler(t_loader)
    v_sampler = DistributedSampler(v_loader, shuffle=False)

    if rank == 0:
        print('#Envs in train: %d' % (len(t_loader.files)))
        print('#Envs in val: %d' % (len(v_loader.files)))

    #########################################################
    ####################### To DO ###########################
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"] // world_size,
        num_workers=cfg["training"]["n_workers"],
        drop_last=True,
        pin_memory=True,
        sampler=t_sampler,
        multiprocessing_context='fork',
    )

    valloader = data.DataLoader(
        v_loader,
        batch_size=cfg["training"]["batch_size"] // world_size,
        num_workers=cfg["training"]["n_workers"],
        pin_memory=True,
        sampler=v_sampler,
        multiprocessing_context='fork',
    )

    #################################################### Setup Model ###################################################

    cfg_model = cfg['model']
    backbone = cfg_model['backbone']
    print('backbone:', backbone)

    if backbone == 'transformer':
        model = BEV360_segformer(cfg_model, device)
    elif backbone == 'segnext':
        model = BEV360_segnext(cfg_model, device)

    model = model.to(device)

    if device.type == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    if rank == 0:
        print('# trainable parameters = ', params)

    # Setup optimizer, lr_scheduler and loss function ##################################################################
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)

    if rank == 0:
        logger.info("Using optimizer {}".format(optimizer))

    lr_decay_lambda = lambda epoch: cfg['training']['scheduler']['lr_decay_rate'] ** (
                epoch // cfg['training']['scheduler']['lr_epoch_per_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    # Setup Metrics
    obj_running_metrics = IoU(cfg['model']['n_obj_classes'])
    obj_running_metrics_val = IoU(cfg['model']['n_obj_classes'])
    obj_running_metrics.reset()
    obj_running_metrics_val.reset()
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    # setup Loss
    loss_fn = SemmapLoss()
    loss_fn = loss_fn.to(device=device)

    if rank == 0:
        logger.info("Using loss {}".format(loss_fn))

    # init training
    start_iter = 0
    start_epoch = 0
    best_iou = -100.0

    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            if rank == 0:
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
                )
                print(
                    "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
                )
            checkpoint = torch.load(cfg["training"]["resume"], map_location="cpu")
            model_state = checkpoint["model_state"]
            model.load_state_dict(model_state)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"]
            start_iter = checkpoint["iter"]
            best_iou = checkpoint['best_iou']
            if rank == 0:
                logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
        else:
            if rank == 0:
                logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))
                print("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    elif cfg['training']['load_model'] is not None:
        checkpoint = torch.load(cfg["training"]["load_model"], map_location="cpu")
        model_state = checkpoint['model_state']
        model.load_state_dict(model_state)
        if rank == 0:
            logger.info("Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["load_model"]))
            print("Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["load_model"]))

    ########################################################################################################
    # start training Loop
    iter = start_iter

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, cfg["training"]["train_epoch"], 1):

        t_sampler.set_epoch(epoch)

        for batch in trainloader:

            iter += 1
            start_ts = time.time()
            rgb, rgb_no_norm, masks_inliers, proj_indices, semmap_gt, map_mask, map_heights = batch

            model.train()
            optimizer.zero_grad()

            semmap_pred, observed_masks = model(rgb, proj_indices, masks_inliers, rgb_no_norm, map_mask, map_heights)

            semmap_gt = semmap_gt.long()

            if observed_masks.any():

                loss = loss_fn(semmap_gt.to(device), semmap_pred, observed_masks)

                with torch.autograd.detect_anomaly():
                    loss.backward()

                optimizer.step()

                semmap_pred = semmap_pred.permute(0, 2, 3, 1)
                masked_semmap_gt = semmap_gt[observed_masks]
                masked_semmap_pred = semmap_pred[observed_masks]

                obj_gt = masked_semmap_gt.detach()
                obj_pred = masked_semmap_pred.data.max(-1)[1].detach()
                obj_running_metrics.add(obj_pred, obj_gt)

            time_meter.update(time.time() - start_ts)

            if (iter % cfg["training"]["print_interval"] == 0):
                conf_metric = obj_running_metrics.conf_metric.conf
                conf_metric = torch.FloatTensor(conf_metric)
                conf_metric = conf_metric.to(device)
                distrib.all_reduce(conf_metric)
                distrib.all_reduce(loss)
                loss /= world_size

                if (rank == 0):
                    conf_metric = conf_metric.cpu().numpy()
                    conf_metric = conf_metric.astype(np.int32)
                    tmp_metrics = IoU(cfg['model']['n_obj_classes'])
                    tmp_metrics.reset()
                    tmp_metrics.conf_metric.conf = conf_metric
                    _, mIoU, acc, _, mRecall, _, mPrecision = tmp_metrics.value()
                    writer.add_scalar("train_metrics/mIoU", mIoU, iter)
                    writer.add_scalar("train_metrics/mRecall", mRecall, iter)
                    writer.add_scalar("train_metrics/mPrecision", mPrecision, iter)
                    writer.add_scalar("train_metrics/Overall_Acc", acc, iter)

                    fmt_str = "Iter: {:d} == Epoch [{:d}/{:d}] == Loss: {:.4f} == mIoU: {:.4f} == mRecall:{:.4f} == mPrecision:{:.4f} == Overall_Acc:{:.4f} == Time/Image: {:.4f}"

                    print_str = fmt_str.format(
                        iter,
                        epoch,
                        cfg["training"]["train_epoch"],
                        loss.item(),
                        mIoU,
                        mRecall,
                        mPrecision,
                        acc,
                        time_meter.avg / cfg["training"]["batch_size"],
                    )

                    print(print_str)
                    writer.add_scalar("loss/train_loss", loss.item(), iter)
                    time_meter.reset()

        ########## validation ###########
        model.eval()
        with torch.no_grad():
            for batch_val in valloader:

                rgb, rgb_no_norm, masks_inliers, proj_indices, semmap_gt, map_mask, map_heights  = batch_val
                # semantic = semantic.squeeze(0).to(device)


                semmap_pred, observed_masks = model(rgb, proj_indices, masks_inliers, rgb_no_norm, map_mask, map_heights)
                semmap_gt = semmap_gt.long()

                if observed_masks.any():
                    loss_val = loss_fn(semmap_gt.to(device), semmap_pred, observed_masks)

                    #####
                    semmap_pred = semmap_pred.permute(0, 2, 3, 1)

                    masked_semmap_gt = semmap_gt[observed_masks]
                    masked_semmap_pred = semmap_pred[observed_masks]


                    obj_gt_val = masked_semmap_gt
                    obj_pred_val = masked_semmap_pred.data.max(-1)[1]
                    obj_running_metrics_val.add(obj_pred_val, obj_gt_val)

                    val_loss_meter.update(loss_val.item())

        conf_metric = obj_running_metrics_val.conf_metric.conf
        conf_metric = torch.FloatTensor(conf_metric)
        conf_metric = conf_metric.to(device)
        distrib.all_reduce(conf_metric)

        val_loss_avg = val_loss_meter.avg
        val_loss_avg = torch.FloatTensor([val_loss_avg])
        val_loss_avg = val_loss_avg.to(device)
        distrib.all_reduce(val_loss_avg)
        val_loss_avg /= world_size

        if rank == 0:
            val_loss_avg = val_loss_avg.cpu().numpy()
            val_loss_avg = val_loss_avg[0]
            writer.add_scalar("loss/val_loss", val_loss_avg, iter)

            logger.info("Iter %d Loss: %.4f" % (iter, val_loss_avg))

            conf_metric = conf_metric.cpu().numpy()
            conf_metric = conf_metric.astype(np.int32)
            tmp_metrics = IoU(cfg['model']['n_obj_classes'])
            tmp_metrics.reset()
            tmp_metrics.conf_metric.conf = conf_metric
            _, mIoU, acc, _, mRecall, _, mPrecision = tmp_metrics.value()
            writer.add_scalar("val_metrics/mIoU", mIoU, iter)
            writer.add_scalar("val_metrics/mRecall", mRecall, iter)
            writer.add_scalar("val_metrics/mPrecision", mPrecision, iter)
            writer.add_scalar("val_metrics/Overall_Acc", acc, iter)

            logger.info("val -- mIoU: {}".format(mIoU))
            logger.info("val -- mRecall: {}".format(mRecall))
            logger.info("val -- mPrecision: {}".format(mPrecision))
            logger.info("val -- Overall_Acc: {}".format(acc))

            print("val -- mIoU: {}".format(mIoU))
            print("val -- mRecall: {}".format(mRecall))
            print("val -- mPrecision: {}".format(mPrecision))
            print("val -- Overall_Acc: {}".format(acc))

            if mIoU >= best_iou:
                best_iou = mIoU
                state = {
                    "epoch": epoch,
                    "iter": iter,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_iou,
                }
                save_path = os.path.join(
                    writer.file_writer.get_logdir(),
                    "{}_mp3d_best_model.pkl".format(cfg["model"]["arch"]),
                )
                torch.save(state, save_path)

                # -- save checkpoint after best epoch
                state = {
                    "epoch": epoch,
                    "iter": iter,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_iou,
                }
                save_path = os.path.join(cfg['checkpoint_dir'], "ckpt_model.pkl")
                torch.save(state, save_path)

        val_loss_meter.reset()
        obj_running_metrics_val.reset()
        obj_running_metrics.reset()

        scheduler.step(epoch)


########################################################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/model_360BEV_mp3d.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    name_expe = cfg['name_experiment']

    run_id = random.randint(1, 100000)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    run_id = nowTime

    logdir = os.path.join("runs", name_expe, str(run_id))
    chkptdir = os.path.join("checkpoints", name_expe, str(run_id))

    cfg['checkpoint_dir'] = chkptdir
    cfg['logdir'] = logdir

    print("RUNDIR: {}".format(logdir))
    Path(logdir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, logdir)

    print("CHECKPOINTDIR: {}".format(chkptdir))
    Path(chkptdir).mkdir(parents=True, exist_ok=True)

    world_size = 4
    mp.spawn(train,
             args=(world_size, cfg),
             nprocs=world_size,
             join=True)
