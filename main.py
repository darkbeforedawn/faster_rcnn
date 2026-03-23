from pprint import pprint

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VOCDataset, rpn_collate_fn
from loss import LossFn
from model import FasterRCNN
from metrics import mean_avg_precision, recall_at_k_threshold


EPOCHS = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def batch_gd(model, loader, opt, scaler_fn, loss_fn, sched_fn):
    """training loop fn"""
    model.train()
    step = 0
    pbar = tqdm(total=EPOCHS, desc="Training RPN")

    while step < EPOCHS:
        for batch_list in loader:
            if step >= EPOCHS:
                break

            opt.zero_grad(set_to_none=True)
            batch_loss = 0
            batch_rpn_loss = 0
            batch_roi_loss = 0

            # Loop through every image in the batch
            for batch in batch_list:
                img = batch['image'].to(DEVICE).unsqueeze(0)
                bboxes = batch['bboxes'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                    # Forward pass for this specific image
                    out = model(x=img, target_bbox=bboxes, target_cls=labels)
                    # Calculate loss for THIS image
                    loss_results = loss_fn(out=out)
                # Accumulate the loss (divided by batch size to keep scale consistent)
                # Note: PyTorch keeps the gradient graph alive for each 'out'
                scaler_fn.scale(loss_results['total_loss'] / len(batch_list)).backward()
                # For logging
                batch_loss += loss_results['total_loss'].item()
                batch_rpn_loss += loss_results['rpn_loss']
                batch_roi_loss += loss_results['roi_loss']
            # Step once per batch
            # optimizer.step()
            scaler_fn.step(opt)
            scaler_fn.update()
            sched_fn.step()
            # 3. Log the average batch results
            if step % 1000 == 0:
                print(
                    f"STEP: {step+1} | Loss: {batch_loss/len(batch_list):.4f} | "
                    f"RPN: {batch_rpn_loss/len(batch_list):.4f} | ROI: {batch_roi_loss/len(batch_list):.4f} | "
                    f"LR: {sched_fn.get_last_lr()[0]:.6f}"
                    )
            step += 1
            pbar.update(1)
    pbar.close()

if __name__ == '__main__':

    train_ds = VOCDataset('data/pascal_voc/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007', 'train')
    val_ds = VOCDataset('data/pascal_voc/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007', 'val')

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, pin_memory=True,
        collate_fn=rpn_collate_fn, num_workers=2
        )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=True, pin_memory=True,
        collate_fn=rpn_collate_fn, num_workers=2
        )

    fcnn = FasterRCNN(pre_nms_filter=10000, post_nms_topk=2000).to(DEVICE)

    for m in fcnn.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    criterion = LossFn(alpha=1, beta=1.)
    optimizer = torch.optim.SGD(
        [p for p in fcnn.parameters() if p.requires_grad],
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS, 5e-5
    )
    scaler = torch.amp.GradScaler("cuda")

    batch_gd(fcnn, train_loader, optimizer, scaler, criterion, scheduler)

    recall_at_k_threshold(
        model=fcnn.rpn,
        loader=val_loader,
        device=DEVICE,
        k=(100, 300, 1000),
        iou_t=(0.5, 0.7)
        )

    r = mean_avg_precision(
        model=fcnn,
        loader=val_loader,
        device=DEVICE
    )
    pprint(r)
