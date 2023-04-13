import torch
from torch import nn
import torch.nn.functional as F


class SemmapLoss(nn.Module):
    def __init__(self):
        super(SemmapLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, obj_gt, obj_pred, mask):
        mask = mask.float()
        loss = self.loss(obj_pred, obj_gt)
        loss = torch.mul(loss, mask)
        # -- mask is assumed to have a least one value
        # print('mask_in_loss:', mask.sum(), mask)

        loss = loss.sum()/mask.sum()
        # print("loss_loss:", loss)

        return loss

class hcl_loss_masked(nn.Module):
    def __init__(self):
        super(hcl_loss_masked, self).__init__()

    def forward(self, fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            n,c,h,w = fs.shape
            # print('fs_shape:', fs.size(), ft.size())

            fs_masked = fs[:, :, round(0.4*h):round(0.6*h), :] 
            ft_masked = ft[:, :, round(0.4*h):round(0.6*h), :] 
            # print('fs_masked:', fs_masked.size())

            loss = F.mse_loss(fs_masked, ft_masked, reduction='mean')
            # print('loss_in_mid:', loss.size(), loss)
            ##################################################################################################################################
            # mask = torch.zeros(( h, w), device=loss.device)
            # mask[round(0.3*h): round(0.7*h), :] = 1
            # loss = torch.mul(loss, mask)
            # # -- mask is assumed to have a least one value
            # # print('mask_in_loss:', mask.sum(), loss.sum(), loss)
            # loss = loss.sum()/mask.sum()/c
            
            ##################################################################################################################################
            
            cnt = 1.0
            tot = 1.0
            for l in [4,2,1]:
                if l >=h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs_masked, (l,l))
                tmpft = F.adaptive_avg_pool2d(ft_masked, (l,l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            # print('loss_end:', loss, tot)
            loss_all = loss_all + loss
        return loss_all
    ###############################################################################################################################################
class hcl_loss(nn.Module):
    def __init__(self):
        super(hcl_loss, self).__init__()

    def forward(self, fstudent, fteacher):
        loss_all = 0.0
        if isinstance(fstudent, list) == True:

            for fs, ft in zip(fstudent, fteacher):
                n,c,h,w = fs.shape
                # print('fs,ft:', fs.size())
                loss = F.mse_loss(fs, ft, reduction='mean')

                cnt = 1.0
                tot = 1.0
                for l in [4,2,1]:
                    if l >=h:
                        continue
                    tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                    tmpft = F.adaptive_avg_pool2d(ft, (l,l))
                    cnt /= 2.0
                    loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                    tot += cnt
                loss = loss / tot
                loss_all = loss_all + loss
        else:
            fs, ft = fstudent, fteacher

            n,c,h,w = fs.shape
            loss = F.mse_loss(fs, ft, reduction='mean')
            
            cnt = 1.0
            tot = 1.0
            for l in [4,2,1]:
                if l >=h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                tmpft = F.adaptive_avg_pool2d(ft, (l,l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all




################################################################################################################################################
################################################################################################################################################
class KL_div_loss(nn.Module):
    def __init__(self):
        super(KL_div_loss, self).__init__()
        # self.loss = F.kl_div(reduction="none")

    def forward(self, fstudent, fteacher):
        loss_all = 0.0

        # mask = mask.float()
        for fs, ft in zip(fstudent, fteacher):
            n,c,h,w = fs.shape
            # print('fs_shape:', fs.size(), ft.size())

            loss = F.kl_div(fs.softmax(dim=-1).log(), ft.softmax(dim=-1), reduction="none")
            # print('loss_in_kl:', loss, loss.size())

            
            mask = torch.zeros(( h, w), device=loss.device)
            mask[round(0.3*h): round(0.7*h), :] = 1
        
            # print('mask:', mask.device)

            loss = torch.mul(loss, mask)
            # -- mask is assumed to have a least one value
            # print('mask_in_loss:', mask.sum(), mask)

            loss = loss.sum()/mask.sum()
            loss_all = loss_all + loss
            # print("loss_loss:", loss_all, mask.sum())
 
        return loss_all


#############################################################################################################################################################

class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap



class CriterionCWD(nn.Module):

    def __init__(self,norm_type='channel',divergence='kl',temperature=1.0):
    
        super(CriterionCWD, self).__init__()
       

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x:x.view(x.size(0),x.size(1),-1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = temperature

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

    def forward(self, fstudent, fteacher):

        loss_all = 0.0
        # for fs, ft in zip(fstudent, fteacher):
        # fs = fstudent[3]
        # ft = fteacher[3]
        fs = fstudent
        ft = fteacher


        n,c,h,w = fs.shape

        fs = fs[:, :, round(0.4*h):round(0.7*h), :] 
        ft = ft[:, :, round(0.4*h):round(0.7*h), :] 

        #import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(fs/self.temperature)
            norm_t = self.normalize(ft.detach()/self.temperature)
        else:
            norm_s = fs[0]
            norm_t = ft[0].detach()
        
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s,norm_t)
        
        #item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        #import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        loss_all = loss_all + loss
        return loss_all * (self.temperature**2)
        # return loss * (self.temperature**2)


    # def forward(self,fstudent, fteacher):

    #     loss_all = 0.0
    #     for fs, ft in zip(fstudent, fteacher):

    #         n,c,h,w = fs.shape
    #         #import pdb;pdb.set_trace()
    #         if self.normalize is not None:
    #             norm_s = self.normalize(fs/self.temperature)
    #             norm_t = self.normalize(ft.detach()/self.temperature)
    #         else:
    #             norm_s = fs[0]
    #             norm_t = ft[0].detach()
            
            
    #         if self.divergence == 'kl':
    #             norm_s = norm_s.log()
    #         loss = self.criterion(norm_s,norm_t)
            
    #         #item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
    #         #import pdb;pdb.set_trace()
    #         if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
    #             loss /= n * c
    #             # loss /= n * h * w
    #         else:
    #             loss /= n * h * w

    #         loss_all = loss_all + loss
    #     return loss_all * (self.temperature**2)
    #     # return loss * (self.temperature**2)