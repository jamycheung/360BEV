import yaml
from torch.utils import data
from metric.iou import IoU
import cv2
import numpy as np
import torch.nn
from pathlib import Path

# from model.trans4pano_map import Trans4map_segformer
# from model.trans4pano_deformable_detr import Trans4map_deformable_detr
from model.front_view_segformer_matterport import front_view_segformer
from model.Attention360_pano_matterport import Attention360_pano
from utils.semantic_utils import color_label

# from model.pano_data_loader import DatasetLoader_pano
# from torch.utils.data import DistributedSampler
from utils.lib2_mp3d.dataset import matterport_SemDataset33

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################################################
# model_path = "./checkpoints/model_pano_segformer/2023-02-09-00-40-B2-crop-distortion/ckpt_model.pkl"
# model_path = "./checkpoints/model_pano_segformer/2023-02-08-23-30-B4-crop/ckpt_model.pkl"
# model_path = "./checkpoints/model_pano_segformer/2023-02-18-18-49/ckpt_model.pkl"  # 360Attention
# model_path = "./checkpoints/model_pano_segformer/2023-03-13-20-57/ckpt_model.pkl"  # tras4pass
########################################################################################################################
config_path = "configs/model_fv_mp3d.yml"

with open(config_path) as fp:
    cfg = yaml.safe_load(fp)
########################################################################################################################
output_dir = cfg['output_dir']
Path(output_dir).mkdir(parents=True, exist_ok=True)

cfg_model = cfg['model']

###### init model
model = Attention360_pano(cfg_model, device) # 360Attention
# model = front_view_segformer(cfg_model, device)  # trans4pass+
model = model.to(device)

model_path = cfg['model_path']
print('Loading pre-trained weights:', model_path)

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
###########################################################################################

test_loader = matterport_SemDataset33(cfg["data"], split=cfg["data"]["val_split"])

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
        # rgb, rgb_no_norm, masks_inliers, proj_indices, semmap_gt, map_mask, map_heights = batch
        rgb, semmap_gt, fname= batch

        rgb = rgb.to(device)
        observed_masks = (semmap_gt >= 0)
        semmap_gt[~observed_masks] = 0
        semmap_gt = semmap_gt.long()

        semmap_pred, observed_mask  = model(rgb, observed_masks)


        if observed_masks.any():

            semmap_pred = semmap_pred.permute(0,2,3,1)

            ############################################################################################################
            pred = semmap_pred[observed_masks].softmax(-1)
            pred = torch.argmax(pred, dim = 1).cpu()

            num_classes = 20
            gt = semmap_gt[observed_masks]

            assert gt.min() >= 0 and gt.max() < num_classes and semmap_pred.shape[3] == num_classes
            cm += np.bincount((gt * num_classes + pred).cpu().numpy(), minlength=num_classes**2)

            ############################################################################################################

            semmap_pred_write  = semmap_pred.data.max(-1)[1] + 1

            semmap_pred_write[~observed_mask] = 0
            semmap_pred_write = semmap_pred_write.squeeze(0)

            ############################ semmap projection to show #####################################################
            semmap_pred_write_out = color_label(semmap_pred_write).squeeze(0)
            semmap_pred_write_out = semmap_pred_write_out.permute(1, 2, 0)
            semmap_pred_write_out = semmap_pred_write_out.cpu().numpy().astype(np.uint8)
            semmap_pred_write_out = cv2.cvtColor(semmap_pred_write_out, cv2.COLOR_BGR2RGB)
            file_name = fname[0]
            ################################################# test on semmap pred_write_out ############################

            masked_semmap_gt = semmap_gt[observed_mask]
            masked_semmap_pred = semmap_pred[observed_mask]

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
## Summarize_haha
print('  Summarize_hohonet  '.center(50, '='))
cm = cm.reshape(num_classes, num_classes)
# id2class = np.array(valid_dataset.ID2CLASS)
id2class = ['wall', 'floor', 'chair', 'door', 'table', 'picture', 'furniture', 'objects', 'window', 'sofa', 'bed', 'sink', 'stairs', 'ceiling', 'toilet', 'mirror', 'shower', 'bathtub', 'counter', 'shelving']
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
# np.savez(os.path.join(args.out, 'cm.npz'), cm=cm)


