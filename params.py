from utils.param import  Param


PARAM=Param()

PARAM.dataset_train = Param(
    root=r"G:\数据集\Weakly Supervised Learning for Industrial Optical Inspection\Class1",
                      data_config="class1_train5valid5",
                      phase="training",
                      return_numpy=True)
PARAM.dataloader_train = Param(batch_size=1,
                         shuffle=True,
                         num_workers=8,
                         drop_last=False)

PARAM.dataset_valid = Param(
    root=r"G:\数据集\Weakly Supervised Learning for Industrial Optical Inspection\Class1",
                      data_config="class1_train5valid5",
                      phase="validation",
                      return_numpy=True)

PARAM.dataloader_valid = Param(batch_size=1,
                         shuffle=False,
                         num_workers=8,
                         drop_last=False)

PARAM.model = Param(n_classes=1, level=4,b_RGB=False,base_channels=16) # 0-1值

PARAM.Adam = Param(
    lr=0.001,
    weight_decay=0.001,
    betas=(0.9, 0.999))

PARAM.train=Param(
    epoch=100,
    valid_frequency=1,
)