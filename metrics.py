from typing import Tuple, Union, Any
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


@torch.inference_mode()
def recall_at_k_threshold(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Union[str, Any],
    k: Tuple[int]=(100, 300, 1000),
    iou_t: Tuple[float]=(0.5, 0.7)
    ):
    """
        to evaluate our RPN, we utilize Recall@k[t] where k are the number of top-k anchor proposals and t is the iou threshold
        we first filter the top-k bbox final proposals after regression + filtering + NMS
        for each gt-box we measure the iou with those top-k rpn proposals
        for each GT box, check whether any of the top-k rpn proposals matches it above IoU threshold
        we select the BEST iou for each GTbox and check if that is >= t
        if it is >= t we count it as recall
        recall is the num iou>t / num_gt box
        we do this for Recall@(100, 300, 1000)[0.5, 0.7] -> giving us 3*2=6 metrics results
    """
    model.eval()
    for top_k in k:
        for thresh in iou_t:
            correct, total = 0, 0
            for batch in tqdm(loader, total=len(loader)):
                for row in batch:
                    # img shape [C, H, W]; gt_box shape list[[4]]
                    img = row['image'].unsqueeze(0).to(device)
                    out = model(img, None)
                    idx = torch.argsort(out['cls_obj_scores'][0], dim=-1, descending=True)
                    # filter out the top-k indices
                    idx = idx[:top_k]
                    # top_k_p -> tensor of shape [N, 4]
                    top_k_p: torch.Tensor = out['proposals'][0][idx]
                    # boxes have shape [M, 4] where N >> M
                    boxes = torch.as_tensor(row['bboxes'], dtype=torch.float32, device=top_k_p.device)
                    if boxes.numel() == 0:
                        continue
                    # this has shape [N, M] for each N(row)proposal M(col)gt-box ious
                    iou_mat = torchvision.ops.box_iou(top_k_p, boxes)
                    # max_iou has shape [M] the max-iou per (M) GT-box
                    max_iou, _ = torch.max(iou_mat, dim=0)
                    correct += (max_iou >= thresh).sum().item()
                    total += max_iou.numel()
            recall = correct / total
            print(f"Recall@{top_k} with IoU@{thresh} is: {recall:.5f}")

@torch.inference_mode()
def mean_avg_precision(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Union[str, Any],
    ):
    """
    calculate mean avg precision (mAP)

    Args:
        model (torch.nn.Module): model
        loader (torch.utils.data.DataLoader): validation/test loader
        device (Union[str, Any]): device

    Returns:
        _type_: results map
    """
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        iou_thresholds=[0.5],
        class_metrics=True,
    )
    model.eval()
    for batch in loader:
        for row in batch:
            img = row['image'].unsqueeze(0).to(device)
            out = model(img, None, None)
            preds = [{
                "boxes": out['boxes'].to(device),
                "scores": out['scores'].to(device),
                "labels": out['labels'].to(device),
                }]
            targets = [{
                "boxes": row['bboxes'].to(device),
                "labels": row['labels'].to(device),
            }]
            metric.update(preds, targets)
    results = metric.compute()
    metric.reset()
    return results
