import yaml
from torch.utils import data
from metric.iou import IoU
import cv2
import numpy as np
import torch.nn
from pathlib import Path

# from model.front_view_segformer import front_view_segformer
from model.Attention360_pano_s2d3d import Attention360_pano_s2d3d
from utils.semantic_utils import color_label
from utils.lib2_s2d3d.dataset.dataset_s2d3d_sem_class13 import S2d3dSemDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################################################################
# # model_path = "./checkpoints/model_pano_segformer/2023-03-06-22-16/ckpt_model.pkl"
# # model_path = "./checkpoints/model_pano_segformer/2023-03-07-16-55/ckpt_model.pkl"
# model_path = "./checkpoints/model_pano_segformer/2023-03-07-22-45/ckpt_model.pkl" # ckpt for 360Attention
# # model_path = "./checkpoints/model_pano_segformer/2023-02-25-16-54/ckpt_model.pkl" # trans4pass test
########################################################################################################################
# state = torch.load(model_path, map_location='cpu')
########################################################################################################################
config_path = "configs/model_fv_s2d3d.yml"

with open(config_path) as fp:
    cfg = yaml.safe_load(fp)

########################################################################################################################
output_dir = cfg['output_dir']
Path(output_dir).mkdir(parents=True, exist_ok=True)

cfg_model = cfg['model']
# backbone = cfg_model['backbone']
# print('backbone:', backbone)
num_classes = cfg_model['n_obj_classes']


####### init model
model = Attention360_pano_s2d3d(cfg_model, device)  ## for 360Attention test
# model = front_view_segformer(cfg_model, device)  ## for tras4pass test
model = model.to(device)

model_path = cfg['model_path']
print('Loading pre-trained weights: ', model_path)

state = torch.load(model_path)
print("best_iou:", state['best_iou'])
model_state = state['model_state']
print('model_state:', model_state.keys())


weights = {}
for k, v in model_state.items():
    k = '.'.join(k.split('.')[1:])
    weights[k] = v

model.load_state_dict(weights)
model.eval()
########################################################################################################################

test_loader = S2d3dSemDataset(cfg["data"], Split=cfg["data"]["val_split"])

testingloader = data.DataLoader(
        test_loader,
        batch_size = 1,
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

        rgb, semmap_gt, fname= batch

        rgb = rgb.to(device)
        observed_masks = (semmap_gt >= 0)
        semmap_gt[~observed_masks] = 0
        semmap_gt = semmap_gt.long()

        # semmap_pred, observed_masks, rgb_write = model(rgb, proj_indices, masks_inliers, rgb_no_norm)
        # semmap_pred, observed_masks = model(rgb, proj_indices, masks_inliers, rgb_no_norm, map_mask, map_heights)
        semmap_pred, observed_mask  = model(rgb, observed_masks)

        if observed_masks.any():

            semmap_pred = semmap_pred.permute(0,2,3,1)

            ############################################################################################################
            pred = semmap_pred[observed_masks].softmax(-1)
            pred = torch.argmax(pred, dim = 1).cpu()

            # num_classes = 13
            gt = semmap_gt[observed_masks]

            assert gt.min() >= 0 and gt.max() < num_classes and semmap_pred.shape[3] == num_classes
            cm += np.bincount((gt * num_classes + pred).cpu().numpy(), minlength=num_classes**2)

            ############################################################################################################
            semmap_pred_write  = semmap_pred.data.max(-1)[1] + 1

            semmap_pred_write[~observed_mask] = 0
            semmap_pred_write = semmap_pred_write.squeeze(0)

            ############################ semmap projection to show #####################################################
            # ###############################semmap_gt to show #########################################################
            semmap_gt_write = semmap_gt + 1
            semmap_gt_write[~observed_mask] = 0
            semmap_gt_write = semmap_gt_write.squeeze(0)

            semmap_gt_write_out = color_label(semmap_gt_write).squeeze(0)
            semmap_gt_write_out = semmap_gt_write_out.permute(1,2,0)
            semmap_gt_write_out = semmap_gt_write_out.cpu().numpy().astype(np.uint8)
            semmap_gt_write_out = cv2.cvtColor(semmap_gt_write_out, cv2.COLOR_BGR2RGB)
            file_name = fname[0]
            #####################################################################################
            ###############################semmap projection mask to show #######################
            #################################### RGB_To_Show ###########################################################
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

#########################################################################################################################################
## Summarize_haha
print('  Summarize_hohonet  '.center(50, '='))
cm = cm.reshape(num_classes, num_classes)
# id2class = np.array(valid_dataset.ID2CLASS)
id2class = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
id2class = np.array(id2class)

valid_mask = (cm.sum(1) != 0)
print('valid_mask:', valid_mask)
cm = cm[valid_mask][:, valid_mask]
id2class = id2class[valid_mask]

inter = np.diag(cm)
union = cm.sum(0) + cm.sum(1) - inter
ious = inter / union
recalls = inter / cm.sum(1)
precisions =  inter / cm.sum(0)
accs = np.sum(inter) / np.sum(cm)

for name, iou, recall, precision in zip(id2class, ious, recalls, precisions):
    print(f'{name:20s}:    iou {iou*100:5.2f}    / recall {recall*100:5.2f}   / precision {precision*100:5.2f}')
print(f'{"Overall":20s}:   iou {ious.mean()*100:5.2f}  / recall {recalls.mean()*100:5.2f}   / precision {precisions.mean()*100:5.2f}  /    acc {accs*100:5.2f}')


