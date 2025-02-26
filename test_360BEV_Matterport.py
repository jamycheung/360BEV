import yaml
import numpy as np
import torch.nn
from pathlib import Path
import argparse


from torch.utils import data
from metric.iou import IoU
from model.BEV360_segformer_matterport import BEV360_segformer
from model.BEV360_segnext_matterport import BEV360_segnext

from utils.semantic_utils import color_label
from model.dataloader_mp3d.pano_data_loader import DatasetLoader_pano_detr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################################################################
# config_path = "configs/model_360BEV_mp3d.yml"

######################
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

########################################################################################################################
output_dir = cfg['output_dir']
Path(output_dir).mkdir(parents=True, exist_ok=True)

cfg_model = cfg['model']
backbone = cfg_model['backbone']
print('backbone:', backbone)
num_classes = cfg_model['n_obj_classes']

if backbone == 'transformer':
    model = BEV360_segformer(cfg_model, device)
elif backbone == 'segnext':
    model = BEV360_segnext(cfg_model, device)

model = model.to(device)

model_path = cfg['model_path']
print('Loading pre-trained weights: ', model_path)
state = torch.load(model_path)
print("best_iou:", state['best_iou'])
model_state = state['model_state']

weights = {}
for k, v in model_state.items():
    if k == 'module.reference_points.weight' or k == 'module.reference_points.bias':
        continue
    k = '.'.join(k.split('.')[1:])
    weights[k] = v

model.load_state_dict(weights)
model.eval()

########################################################################################################################
########################################################################################################################

test_loader = DatasetLoader_pano_detr(cfg["data"], split=cfg["data"]["val_split"])

testingloader = data.DataLoader(
        test_loader,
        batch_size=1,
        num_workers=cfg["training"]["n_workers"],
        pin_memory=True,
        # sampler=test_sampler,
        multiprocessing_context='fork',
    )

##### setup Metrics #####
obj_running_metrics_test = IoU(cfg['model']['n_obj_classes'])
cm = 0

with torch.no_grad():
    for batch in testingloader:

        # rgb, rgb_no_norm, masks_inliers, proj_indices, semmap_gt = batch
        rgb, rgb_no_norm, proj_indices, semmap_gt, map_mask, map_heights = batch

        rgb = rgb.to(device)
        proj_indices = proj_indices.to(device)
        map_heights = map_heights.to(device)
        semmap_gt = semmap_gt.long()
        map_mask = map_mask.to(device)

        # semmap_pred, observed_masks, rgb_write = model(rgb, proj_indices, masks_inliers, rgb_no_norm)
        semmap_pred, observed_masks = model(rgb, proj_indices, rgb_no_norm, map_mask, map_heights)

        if observed_masks.any():
            semmap_pred = semmap_pred.permute(0,2,3,1)
            ############################################################################################################
            pred = semmap_pred[observed_masks].softmax(-1)
            pred = torch.argmax(pred, dim = 1).cpu()
            pred = pred

            gt = semmap_gt[observed_masks]
            assert gt.min() >= 0 and gt.max() < num_classes and semmap_pred.shape[3] == num_classes
            cm += np.bincount((gt * num_classes + pred).cpu().numpy(), minlength=num_classes**2)
            ############################################################################################################

            semmap_pred_write  = semmap_pred.data.max(-1)[1]
            semmap_mask_write22 = semmap_pred_write
            semmap_pred_write[~observed_masks] = 0
            semmap_pred_write = semmap_pred_write.squeeze(0)
            ###############################semmap_gt to show ####################################
            semmap_gt_write = semmap_gt.squeeze(0)
            semmap_gt_write_out = color_label(semmap_gt_write).squeeze(0)
            semmap_gt_write_out = semmap_gt_write_out.permute(1,2,0)
            semmap_gt_write_out = semmap_gt_write_out.cpu().numpy().astype(np.uint8)

            #################################### RGB_To_Show ################################################

            masked_semmap_gt = semmap_gt[observed_masks]
            masked_semmap_pred = semmap_pred[observed_masks]

            obj_gt_val = masked_semmap_gt
            obj_pred_val = masked_semmap_pred.data.max(-1)[1]
            obj_running_metrics_test.add(obj_pred_val, obj_gt_val)

conf_metric = obj_running_metrics_test.conf_metric.conf
conf_metric = torch.FloatTensor(conf_metric)
conf_metric = conf_metric.to(device)

conf_metric = conf_metric.cpu().numpy()
conf_metric = conf_metric.astype(np.int32)
tmp_metrics = IoU(cfg['model']['n_obj_classes'])
tmp_metrics.reset()
tmp_metrics.conf_metric.conf = conf_metric
_, mIoU, acc, _, mRecall, _, mPrecision = tmp_metrics.value()

print("val -- mIoU: {}".format(mIoU))
print("val -- mRecall: {}".format(mRecall))
print("val -- mPrecision: {}".format(mPrecision))
print("val -- Overall_Acc: {}".format(acc))

########################################################################################################################
## Summarize
print('  Summarize_hohonet  '.center(50, '='))
cm = cm.reshape(num_classes, num_classes)
id2class = ['void', 'wall', 'floor', 'chair', 'door', 'table', 'picture', 'furniture', 'objects', 'window', 'sofa', 'bed', 'sink', 'stairs', 'ceiling', 'toilet', 'mirror', 'shower', 'bathtub', 'counter', 'shelving']
id2class = np.array(id2class)

valid_mask = (cm.sum(1) != 0)
print('valid_mask:', valid_mask)
cm = cm[valid_mask][:, valid_mask]
id2class = id2class[valid_mask]

inter = np.diag(cm)
union = cm.sum(0) + cm.sum(1) - inter
ious = inter / union
accs = inter / cm.sum(1)

for name, iou, acc in zip(id2class, ious, accs):
    print(f'{name:20s}:    iou {iou*100:5.2f}    /    acc {acc*100:5.2f}')
print(f'{"Overall":20s}:    iou {ious.mean()*100:5.2f}    /    acc {accs.mean()*100:5.2f}')
