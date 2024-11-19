############################################################################################################
# 1. 使用argparse类实现可以在训练的启动命令中指定超参数
# 2. 可以通过在启动命令中指定 --seed 来固定网络的初始化方式，以达到结果可复现的效果
# 3_1. 使用了更高级的学习策略 cosine warm up：在训练的前几个epoch使用学习率预热策略，逐步增大学习率从一个很小的值到预设的最大学习率,
#    在预热阶段结束后，使用余弦退火策略逐步减小学习率。在训练开始时，通过预热策略避免了模型不稳定的问题, 
#    而在训练后期，通过余弦退火策略细致地调整模型参数，有助于模型收敛
# 3_2. cosine warm up：在训练的第一轮使用一个较小的lr（warm_up），从第二个epoch开始，随训练轮数逐渐减小lr。
# 4. 可以通过在启动命令中指定 --model 来选择使用的模型 
# 5. 使用amp包实现半精度训练，在保证准确率的同时尽可能的减小训练成本
# 6. 实现了数据加载类的自定义实现
# 7. 可以通过在启动命令中指定 --tensorboard 来进行tensorboard可视化, 默认不启用。
#    注意，使用tensorboad之前需要使用命令 "tensorboard --logdir= log_path"来启动，结果通过网页 http://localhost:6006/'查看可视化结果
############################################################################################################
# --model 可选的超参如下：
# alexnet   zfnet   vgg   vgg_tiny   vgg_small   vgg_big   googlenet   xception   resnet_small   resnet   resnet_big   resnext   resnext_big  
# densenet_tiny   densenet_small   densenet   densenet_big   mobilenet_v3   mobilenet_v3_large   shufflenet_small   shufflenet
# efficient_v2_small   efficient_v2   efficient_v2_large   convnext_tiny   convnext_small   convnext   convnext_big   convnext_huge   vision_transformer1     vision_transformer2    vision_transformer_small   vision_transformer_big   swin_transformer_tiny   swin_transformer_small   swin_transformer

# 训练命令示例： # python train.py --model kansformer1 --num_classes 5
############################################################################################################

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set the GPUs
import argparse
import math
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

import classic_models
from utils.lr_methods import warmup
from dataload.dataload_five_flower import Five_Flowers_Load
from utils.train_engin import train_one_epoch, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=8, help='the number of classes')  # 分类数量
parser.add_argument('--epochs', type=int, default=100, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=40, help='batch_size for training')
# parser.add_argument('--lr', type=float, default=0.0002, help='star learning rate')
parser.add_argument('--lr', type=float, default=0.0001, help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.0001, help='end learning rate')
parser.add_argument('--seed', default=False, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization')
parser.add_argument('--use_amp', default=False, action='store_true',
                    help=' training with mixed precision')  # 是否使用混合精度训练
parser.add_argument('--data_path', type=str, default=r"./data/PBC")
parser.add_argument('--model', type=str, default="efficient_v2", help=' select a model for training')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
# parser.add_argument('--weights', type=str, default=r'model_pth\vit_base_patch16_224.pth', help='initial weights path')
parser.add_argument('--weights', type=str, default=r'', help='initial weights path')

opt = parser.parse_args()

if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed)  # Python random module.
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)  # Numpy module.
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置:
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # 实际上这个设置对精度影响不大，仅仅是小数点后几位的差别。所以如果不是对精度要求极高，其实不太建议修改，因为会使计算效率降低。
        print('random seed has been fixed')


    seed_torch()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    if opt.tensorboard:
        # 这是存放你要使用tensorboard显示的数据的绝对路径
        log_path = os.path.join('./results/tensorboard/PBC', args.model)
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path))

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path)  # 当log文件存在时删除文件夹。记得在代码最开始import shutil

        # 实例化一个tensorboard
        tb_writer = SummaryWriter(log_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 对标pytorch封装好的ImageFlolder，我们自己实现了一个数据加载类 Five_Flowers_Load，并使用指定的预处理操作来处理图像，结果会同时返回图像和对应的标签。  
    train_dataset = Five_Flowers_Load(os.path.join(args.data_path, 'train'), transform=data_transform["train"])
    val_dataset = Five_Flowers_Load(os.path.join(args.data_path, 'val'), transform=data_transform["val"])

    if args.num_classes != train_dataset.num_class:
        raise ValueError("dataset have {} classes, but input {}".format(train_dataset.num_class, args.num_classes))

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 使用 DataLoader 将加载的数据集处理成批量（batch）加载模式
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                               num_workers=0, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                             num_workers=0, collate_fn=val_dataset.collate_fn)

    # create model
    model = classic_models.find_model_using_name(opt.model, num_classes=opt.num_classes).to(device)
    # 若需要加载预训练权重将下面这段注释打开即可
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.layers.1.base_weight', 'head.layers.1.spline_weight','head.layers.1.spline_scaler']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    best_acc = 0.

    # save parameters path
    # save_path = os.path.join(os.getcwd(), 'results/weights_pretraind/PBC', args.model)
    save_path = os.path.join(os.getcwd(), 'results/weights/PBC', args.model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for epoch in range(args.epochs):
        # train
        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                               device=device, epoch=epoch, use_amp=args.use_amp, lr_method=warmup)
        scheduler.step()
        # validate
        val_acc, precision, recall, f1_score, b_acc = evaluate(model=model, data_loader=val_loader, device=device, ifIndicators=True, Binary=False)

        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' % (
        epoch + 1, mean_loss, train_acc, val_acc))
        with open(os.path.join(save_path, "kansformer_cell_log.txt"), 'a') as f:
            f.writelines('[epoch %d] train_loss: %.5f  train_acc: %.5f precision:%.5f recall:%.5f f1_score:%.5f val_accuracy: %.5f balance_accuracy: %.5f' % (
            epoch + 1, mean_loss, train_acc * 100, precision * 100, recall * 100, f1_score * 100, val_acc * 100, b_acc * 100) + '\n')

        if opt.tensorboard:
            tags = ["train_loss", "train_acc", "precision", "recall", "f1_score", "val_accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], precision, epoch)
            tb_writer.add_scalar(tags[3], recall, epoch)
            tb_writer.add_scalar(tags[4], f1_score, epoch)
            tb_writer.add_scalar(tags[5], val_acc, epoch)
            tb_writer.add_scalar(tags[6], optimizer.param_groups[0]["lr"], epoch)

        # 判断当前验证集的准确率是否是最大的，如果是，则更新之前保存的权重
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     torch.save(model.state_dict(), os.path.join(save_path, str(epoch)+"ep_kansformer_cell_best.pth"))
        #
        # if epoch % 5 == 0:
        #     torch.save(model.state_dict(), os.path.join(save_path, str(epoch)+"ep_kansformer_cell.pth"))


main(opt)
