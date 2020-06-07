import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from unets.resunet import  Res18_UNet
from WSLDatasets.wsl_dataset import  WSLDataset_split
from params import  PARAM
from funcs import  iter_on_a_batch
from  utils.data import  my_transforms
import utils

###定义数据
transform={"train": my_transforms.ComposeJoint([
     my_transforms.ToPIL(),
     my_transforms.GroupRandomHorizontalFlip(),
     my_transforms.GroupRandomVerticalFlip(),
     my_transforms.GroupResize(size=(128,128)),
 ]),
"valid": my_transforms.ComposeJoint([
     my_transforms.ToPIL(),
     my_transforms.GroupResize(size=(128,128)),
 ])
}
train_data = WSLDataset_split(transform_PIL=transform["train"],**(PARAM.dataset_train))
train_loader = DataLoader(train_data, **(PARAM.dataloader_train))
valid_data = WSLDataset_split(transform_PIL=transform["valid"],**(PARAM.dataset_valid))
valid_loader = DataLoader(train_data, **(PARAM.dataloader_valid))



#定义模型
model = Res18_UNet(**(PARAM.model))

#定义损失函数：
loss=nn.BCELoss()

#定义优化器
parameters = filter(lambda p: p.requires_grad, model.parameters())
optim=torch.optim.Adam(params=parameters,**(PARAM.Adam))

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
#定义一个metrics
metrics=utils.segment_metrics.SegmentationMetric(numClass=2)

epoch=PARAM.train.epoch
valid_frequency=PARAM.train.valid_frequency

if __name__=="__main__":
    for epo  in  range(epoch):
        print("epoch:{}......".format(epo))

        #train
        metrics.reset()
        for cnt_batch,batch  in  enumerate(train_loader):
            result=iter_on_a_batch(batch,model,
                                   losses={"supervise":loss},
                                   optim=optim,
                                   phase="train",device=device)
            #每次迭代打印损失函数
            loss_dict=result["loss"]
            s = "epoch:{},batch:{},lr:{:.4f}".format(epo,cnt_batch,float(optim.state_dict()['param_groups'][0]['lr']))
            for key, val in loss_dict.items():
                s += ",{}:{:.4f}".format(key,float(val))
            print(s)

            #添加结果到metrcis
            img_batch, label_pixel_batch, label_batch, file_name_batch = batch
            metrics.addBatch(result["mask_batch"],label_pixel_batch)


        #
        iou_defect = metrics.clsIntersectionOverUnion(1)
        print("----epoch:{},train,iou:{:.4f}".format(epo,iou_defect))


        #验证
        with_valid=True if epo%valid_frequency==0 else False
        if not with_valid:continue
        metrics.reset()
        for cnt_batch,batch  in  enumerate(train_loader):
            result=iter_on_a_batch(batch,model,
                                   losses={"supervise": loss},
                                   optim=optim,
                                   phase="valid",device=device)
            #添加结果到metrcis
            img_batch, label_pixel_batch, label_batch, file_name_batch = batch
            metrics.addBatch(result["mask_batch"],label_pixel_batch)
        print("----epoch:{},valid,iou:{:.4f}".format(epo,iou_defect))


















    #valid


