文件夹exp中的checkpoint和log分别用于保存权重文件和日志。
model中的classifier_layer定义分类头，hgnn_v2定义超图卷积部分以及网络框架。
                semantic定义语义解耦模块

main.py是单卡训练的main文件
ddp_main.py是后来改写的多卡训练时的main文件，使用的dataloader和载入模型的方式不同，日志保存方式也不同。

                