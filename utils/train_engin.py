import sys
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, matthews_corrcoef
import torch
from tqdm import tqdm

from utils.distrubute_utils import  is_main_process, reduce_value
from utils.lr_methods import warmup
 
def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp=False, lr_method=None):
    """
    训练一个 epoch 的函数。

    Args:
        model (nn.Module): 要训练的模型。
        optimizer (torch.optim.Optimizer): 优化器。
        data_loader (DataLoader): 数据加载器，用于迭代训练数据。
        device (torch.device): 训练设备。
        epoch (int): 当前训练的 epoch 数。
        use_amp (bool, optional): 是否使用自动混合精度训练，默认为 False。
        lr_method (callable, optional): 学习率调度方法，默认为 None。

    Returns:
        tuple: 包含训练损失和准确率的元组。
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    train_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)

    optimizer.zero_grad()
    
    lr_scheduler = None
    if epoch == 0  and lr_method == warmup : 
        warmup_factor = 1.0/1000
        warmup_iters = min(1000, len(data_loader) -1)

        lr_scheduler = warmup(optimizer, warmup_iters, warmup_factor)
    
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    
    # 创建一个梯度缩放标量，以最大程度避免使用fp16进行运算时的梯度下溢 
    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred = model(images.to(device))
            loss = loss_function(pred, labels.to(device))

            pred_class = torch.max(pred, dim=1)[1]
            acc_num += torch.eq(pred_class, labels.to(device)).sum()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += reduce_value(loss, average=True).detach()

        # 在进程中打印平均loss
        if is_main_process():
            info = '[epoch{}]: learning_rate:{:.5f}'.format(
                epoch + 1, 
                optimizer.param_groups[0]["lr"]
            )
            data_loader.desc = info # tqdm 成员 desc
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    
    return train_loss.item() / (step + 1), acc_num.item() / sample_num
        
@torch.no_grad()
def evaluate(model, data_loader, device, ifIndicators=False, Binary=False):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset) 
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    if ifIndicators:
        # 用于存储所有的标签和预测值
        all_labels = []
        all_preds = []
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            pred = model(images.to(device))
            pred_class = torch.max(pred, dim=1)[1]
            sum_num += torch.eq(pred_class, labels.to(device)).sum()

            if ifIndicators:
                # 将标签和预测值添加到列表中
                all_labels.extend(labels.tolist())
                all_preds.extend(pred_class.tolist())

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)
    
    sum_num = reduce_value(sum_num, average=False)
    val_acc = sum_num.item() / num_samples
    
    if ifIndicators:
        # 计算Precision，Recall和F1-score
        if Binary == False:
            precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
            balanced_acc = balanced_accuracy_score(all_labels, all_preds, adjusted=False)
            return val_acc, precision, recall, f1_score, balanced_acc
        else:
            precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
            # 计算MCC
            mcc = matthews_corrcoef(all_labels, all_preds)
            return val_acc, precision, recall, f1_score, mcc
    
    return val_acc
 
