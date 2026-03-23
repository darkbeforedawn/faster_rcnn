from typing import Optional, Tuple, Dict, Any

import torch
from torch import nn
import torchvision

BBOX_REG_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0])

def apply_deltas(anchors, anchor_deltas):
    # convert xyxy anchors to xywh format
    # anchors [h*w*a, 4] anchor deltas [b, h*w*a, 4]
    # convert anchors from xyxy to xywh (w=x2-x1;h=y2-y1)
    # apply_deltas says: starting from the anchor, move and scale like this
    # unscale deltas first
    anchor_deltas = anchor_deltas / BBOX_REG_WEIGHTS.to(anchor_deltas.device)
    aw = anchors[..., 2] - anchors[..., 0]
    ah = anchors[..., 3] - anchors[..., 1]
    # centers x,y; x=x1+w/2 y=y1+h/2
    ax = anchors[..., 0] + (aw * 0.5)
    ay = anchors[..., 1] + (ah * 0.5)
    # shapes of ax/ay/aw/ah = [h*w*a,]
    # x=ax+(dx*aw); y=ay+(dy*ah); w=aw*exp(dw); h=ah*exp(dh)
    x = ax + (anchor_deltas[..., 0] * aw)
    y = ay + (anchor_deltas[..., 1] * ah)
    w = aw * torch.exp(anchor_deltas[..., 2])
    h = ah * torch.exp(anchor_deltas[..., 3])
    # shapes of xywh are [1, h*w*a] each represeents a coordinate
    x1, y1 = x - (w * 0.5), y - (h * 0.5)
    x2, y2 = x + (w * 0.5), y + (h * 0.5)
    delta = torch.stack((x1, y1, x2, y2), dim=-1)
    return delta

def encode_deltas(anchors, matched_gt_bboxes):
    # both in xyxy -> convert to xywh
    # convert raw anchors to xywh
    aw = anchors[..., 2] - anchors[..., 0]
    ah = anchors[..., 3] - anchors[..., 1]
    ax = anchors[..., 0] + (0.5 * aw)
    ay = anchors[..., 1] + (0.5 * ah)
    # convert gtboxes in xyxy to xywh
    gw = matched_gt_bboxes[..., 2] - matched_gt_bboxes[..., 0]
    gh = matched_gt_bboxes[..., 3] - matched_gt_bboxes[..., 1]
    gx = matched_gt_bboxes[..., 0] + (0.5 * gw)
    gy = matched_gt_bboxes[..., 1] + (0.5 * gh)
    # apply-deltas: x=ax+(dx*aw); y=ay+(dy*ah); w=aw*exp(dw); h=ah*exp(dh)
    # apply-deltas tells us anchor + delta = bbox
    # we want inverse where anchor + target = delta an inverse of the above
    # encode_deltas says: to reach this GT box from the anchor, here is the move-and-scale you would need
    # now we have gt already so: gx=ax+(dx*aw); gy=ay+(dy*ah); gw=aw*exp(dw); gh=ah*exp(dh)
    # to get d[xywh]: dx=(gx-ax)/aw; dy=(gy-ay)/ah; dw=log(gw/aw); dh=log(gh/ah)
    dx = (gx - ax) / aw.clamp(min=1e-6)
    dy = (gy - ay) / ah.clamp(min=1e-6)
    dw = torch.log(gw.clamp(min=1e-6) / aw.clamp(min=1e-6))
    dh = torch.log(gh.clamp(min=1e-6) / ah.clamp(min=1e-6))
    deltas = torch.stack((dx, dy, dw, dh), dim=-1)
    return deltas * BBOX_REG_WEIGHTS.to(deltas.device)

def iou(box1: torch.Tensor, box2: torch.Tensor):
    # box1 shape is [h*w*a, 4] and box2 is [M, 4]
    # now assume h*w*a = N; box1: [N, 4], box2: [M, 4]
    # Use None to expand dims for broadcasting: [N, 1, 4] and [1, M, 4]
    # This creates an [N, M] result
    # Calculate top-left and bottom-right of intersection
    x_left = torch.max(box1[:, None, 0], box2[:, 0])
    y_top = torch.max(box1[:, None, 1], box2[:, 1])
    x_right = torch.min(box1[:, None, 2], box2[:, 2])
    y_bottom = torch.min(box1[:, None, 3], box2[:, 3])
    # Intersection area with clamp(min=0)
    intersection = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    # Calculate individual areas
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    # Union: Area1 + Area2 - Intersection
    # Use None on area1 to broadcast [N] to [N, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=1e-6)

class RPN(nn.Module):
    """
    what this basically does:
    1. the model outputs ox (objectness score) and ax (anchor-deltas; that tell us shifts to make in the anchors)
    2. generates static anchors for each feature map position HxW centered at (0, 0)
    inference:
        (our goal is to apply the learned deltas to anchors and filter the best ones)
        3. the deltas (ax; shifts/corrections in anchor positions) are applied to the static anchors
        4. these are the proposals - the model essentially predicts the shifts in the default anchors to cover the objects
        5. these proposals are alot, we filter them by sorted wrt the objectness scores
            - then clamp them to the image (not feature map) height and width
            - then remove any anchors that do not fit the minimum height/width criteria
            - then apply NMS and filter the top-k <final proposals>
        6. we return these final proposals and the corresponding filtered top-k objectness (ox)
    training:
        (our goal is to reverse the above (step.3) such that we know the 'delta' required to get from the anchor to GT bbox)
        3. for each anchor we get its best GTbox such that no GTbox is left without an anchor
        4. this is coupled with labels that tell which anchor->gtbox mapping is background or foreground
        5. for each of these mappings we get the 'delta' required to reach the GT from the anchor
        6. these are alot, so we select a balanced sample of 256 (12pos/128neg) indexes <<randomly>>
        7. 128 positive idx from sampled idxs are used to select the delta predictions (raw ax from step.1) and delta needed from step.5
        8. bbox regression is only trained on these positives
        9. the total pos/neg 256 sampled idxs are used to select raw ox (from step.1) and labels from step.4 to train binary classification
    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        pre_nms_filter: int = 10000,
        post_nms_topk: int = 2000,
        prop_filter_min_size: int = 16,
        prop_filter_score_thresh: float = 0.05,
        ) -> None:
        super().__init__()
        self.scales = [32, 64, 128, 256]
        self.aspect_ratios = [0.5, 1, 2]
        self.stride = 16
        self.keep = pre_nms_filter
        self.topk = post_nms_topk
        self.prop_filter_min_size = prop_filter_min_size
        self.prop_filter_score_thresh = prop_filter_score_thresh
        model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        self.backbone = nn.Sequential(*list(model.children())[:-3]) # (b, 1024, 14, 14) given 224x224
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
            ) # (b, 256, 14, 14)
        # for every feature map we have 12 anchors (3 scales and 3 ratios)
        # for every feature map and 12 anchors we need an objectness (has object or not)
        self.objectness = nn.Conv2d(512, 12, 1, 1, 0)
        # for every anchor (12 total) we need 4 deltas (corrections/nudges of the x1,y1,x2,y2 anchors)
        # named as tx, ty, tw, th (12x4=48) deltas
        self.anchor_pred = nn.Conv2d(512, 48, 1, 1, 0)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor]=None) -> Dict[str, Any]:
        # we assume a batch size of 1 and train with it
        # x shape [1, 3, H, W]; target shape [N, 4] -> N objects in x
        _, _, img_h, img_w = x.shape
        fx = self.backbone(x)
        B, _, H, W = fx.shape
        if B > 1:
            raise NotImplementedError("Batch size of >1 not supported")
        x = self.conv3x3(fx)
        ox = self.objectness(x) # (b, 12, h, w)
        ax = self.anchor_pred(x) # (b, 48, h, w)
        anchors = self.generate_anchors(H, W) # (h*w*a, 4); given a=12
        anchors = anchors.to(x.device)
        # permute ox and ax to match anchor format
        # ox (b, a, h, w) -> (b, h, w, a) -> (b, h*w*a)
        ox = ox.permute(0, 2, 3, 1).reshape(B, -1)
        # ax (b, a*4, h, w) -> (b, a, 4, h, w) -> (b, h, w, a, 4) -> (b, h*w*a, 4)
        ax = ax.view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 4)
        # px (b, h*w*a, 4) here is the corrected-predictions/proposals (xyxy format)
        px = apply_deltas(anchors, ax.detach())
        # filter (pre-nms) clamp and then nms the proposals px then final topk filter
        # returns list of shape [b, :+-topk, 4] where +-topk means its variable
        ox_f, px = self.filter_and_clamp_pred(
            px, ox.detach(), img_h, img_w, self.keep,
            self.topk, min_size=self.prop_filter_min_size,
            score_thresh=self.prop_filter_score_thresh
            )
        rpn_out = {
            "feat_map": fx,
            "proposals": px, #dtype=list
            "cls_obj_scores": ox_f #dtype=list
        }
        if not self.training or target is None:
            return rpn_out
        else:
            labels, matched_gt_bboxes = self.anchor_to_gt(anchors, target)
            # dx is of shape [10800, 4] [h*w*a, 4]
            dx = encode_deltas(anchors, matched_gt_bboxes)
            # pos_idx and sampled_idx are of shape [<=128] and [256]
            pos_idx, _, sampled_idx = self.sample_anchors(labels, 256, 0.5)
            # regression inputs and targets
            # if no pos_idx the returned size will be 
            ax = ax[0, pos_idx]
            dx = dx[pos_idx]
            # classification (object/no-object) inputs and targets
            ox = ox[0, sampled_idx]
            labels = labels[sampled_idx].to(dtype=torch.float32)
            return {
                "feat_map": fx,
                "proposals": px,
                "proposal_scores": ox_f,
                "reg_input": ax,
                "reg_target": dx,
                "cls_input": ox,
                "cls_target": labels
            }

    def generate_anchors(self, h, w):
        # assuminh h, w = 30
        base_anchors = self.static_anchors() # (12, 4)
        shift_x = torch.arange(w, dtype=torch.float32) * self.stride + self.stride / 2 # (30,)
        shift_y = torch.arange(h, dtype=torch.float32) * self.stride + self.stride / 2 # (30,)
        # i,j indexing is row-col (matrix/lin-alg) indexing where row -> y and col -> x such that
        # the resultant indexing is (y, x)
        # for the first 30 elements x should increase and y should remain the same
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        # here (900,) corresponds to a flattened feature_map of 30x30
        shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1) # (900,)
        # however for each fmap loc we need 4 anchors in format of xyxy
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1) # (h * w, 4) (900, 4)
        # but we need 12 scales for each fmap loc and for each scale we need 4 anchors of format xyxy
        # base (1, 12, 4); shifts (900, 1, 4) -> (900, 12, 4)
        anchors = base_anchors.unsqueeze(0) + shifts.unsqueeze(1)
        # flatten to get (h*w*a, 4) format where a = anchor_scales and 4 is the anchor coords
        anchors = anchors.reshape(-1, 4) # (900 * 12, 4) = (10800, 4)
        return anchors

    def static_anchors(self):
        # an image of 480x480 will have a stride 16 and resultant fmap will be 30x30
        # at each pos in 30x30 each feature-cell will correspond to 16 px in the 480x480 img
        # ratio = w/h; area = w*h;
        anchors = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                area = scale ** 2
                # w = sqrt(A/r)
                # h = area/w
                w = scale * (ratio ** 0.5)
                h = scale / (ratio ** 0.5)
                # centered at (0, 0) at each feature-cell location
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2
                anchors.append((x1, y1, x2, y2))
        return torch.tensor(anchors, dtype=torch.float32) # (12, 4)

    def filter_and_clamp_pred(
        self, anchor_pred, ox_cls, img_h, img_w, keep: int=10000,
        topk: int=2000, min_size: int=16, score_thresh: float = 0.05):
        # ox_cls [b, h*w*a]; anchor_pred [b, h*w*a, 4]
        # pre nms filtering wrt the cls-scores of objectness
        sorted_ox, idx = torch.sort(torch.sigmoid(ox_cls), dim=1, descending=True)
        sorted_ox, idx = sorted_ox[:, :keep], idx[:, :keep]
        # clamp the bboxes that are negative and out of bounds to the image boundary
        # note that clamping is done wrt the image h/w and not the feature-map h/w
        # using fancy indexing to select idx first
        B = anchor_pred.shape[0]
        # b_idx has shape [B, 1]; here None adds an extra dim
        b_idx = torch.arange(B, device=anchor_pred.device)[:, None]
        x1, y1 = anchor_pred[b_idx, idx, 0], anchor_pred[b_idx, idx, 1]
        x2, y2 = anchor_pred[b_idx, idx, 2], anchor_pred[b_idx, idx, 3]
        # x1y1x2y2 have shape [b, keep]
        x1, x2 = torch.clamp(x1, 0, img_w), torch.clamp(x2, 0, img_w)
        y1, y2 = torch.clamp(y1, 0, img_h), torch.clamp(y2, 0, img_h)
        # px has shape [b, keep, 4]
        px = torch.stack((x1, y1, x2, y2), dim=2)
        nms_px, nms_ox = [], []
        for i in range(B):
            # px[i] has shape [keep, 4]
            # has shape [keep]
            boxes = px[i]
            s = sorted_ox[i]
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            keep_mask = (ws >= min_size) & (hs >= min_size) & (s >= score_thresh)
            boxes = boxes[keep_mask]
            s = s[keep_mask]

            k = torchvision.ops.nms(boxes, s, 0.7)
            k = k[:topk]
            nms_px.append(boxes[k])
            nms_ox.append(s[k])
        # [b, :+-topk, 4]; [b, :+-topk] here +-topk means nms can be anything between 0 - topk
        # therefore we do not stack and return a list instead
        return nms_ox, nms_px

    def anchor_to_gt(self, anchors, gt_box):
        # anchors shape is [h*w*a, 4] gt_box shape [N, 4]
        # assume M=h*w*a; the iou_mat shape is [M, N]
        # also assume M = 10800 and N = 6
        # so anchors [30x30x12, 4] and gt_box [6, 4] -> assumed for simplicity
        # for each anchor get the highest iou-gtbox; classify into BG/Ignore/FG
        # there might be some GT-boxes with no anchors assigned (although rare)
        # so, for each gtbox get the highest anchor -> get all such anchors that match that iou
        # force those anchors to align
        # for each anchor get highest iou -> resultant shape [10800, 6]
        # early return for if there are no gtboxes
        if gt_box.numel() == 0:
            labels = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)
            matched_gt = torch.zeros_like(anchors)
            return labels, matched_gt
        # iou_mat = iou(anchors, gt_box)
        iou_mat = torchvision.ops.boxes.box_iou(anchors, gt_box)
        # anchor_to_gt_idx will have idx for each gtbox(col) corresponding to each anchor(row)
        max_iou_per_anchor, anchor_to_gt_idx = torch.max(iou_mat, dim=1)
        # we need to return labels (fg->1,bg->0) [10800] and anchor-gt-mapping [10800]
        # classify into bg(-2) fg(1) ignore(-1)
        labels = torch.full_like(anchor_to_gt_idx, -1, dtype=torch.long)
        labels[max_iou_per_anchor >= 0.7] = 1
        labels[max_iou_per_anchor < 0.3] = 0
        # mandatory gt assignment -> some gt boxes might not get assigned to any anchor
        max_iou_per_gt, _ = torch.max(iou_mat, dim=0)
        # find all anchors which share the max iou for a specific gtbox
        # Using torch.where gives us (anchor_indices, gt_indices)
        anchors_with_max_iou, gt_indices_for_max = torch.where(iou_mat == max_iou_per_gt)
        # get anchor indexes where the anchor matches the max-iou for the specified gt 
        anchor_to_gt_idx[anchors_with_max_iou] = gt_indices_for_max
        labels[anchors_with_max_iou] = 1
        # since we have for each anchor a gtbox index, clamp to prevent negative indexing
        # we can now get the corresponding gtboxes -> resulting in shape [10800, 4]
        matched_gt = gt_box.index_select(0, anchor_to_gt_idx)
        # matched_gt contains for each anchor -> a gtbox [tx, ty, tx, ty]
        return labels, matched_gt

    def sample_anchors(self, labels, total_samples: int=256, frac_pos: float=0.5):
        # sample anchor indices with upto 256/2 positives, sample the rest with negatives
        # labels are either 0 or 1 with shape [10800]
        pos_idx = torch.where(labels >= 1)[0]
        neg_idx = torch.where(labels == 0)[0]
        pos_count = min(int(total_samples * frac_pos), pos_idx.numel())
        neg_count = min(total_samples - pos_count, neg_idx.numel())
        if pos_idx.numel() >= pos_count:
            perm = torch.randperm(pos_idx.numel(), device=labels.device)[:pos_count]
            pos_idx = pos_idx[perm]
        if neg_idx.numel() >= neg_count:
            perm = torch.randperm(neg_idx.numel(), device=labels.device)[:neg_count]
            neg_idx = neg_idx[perm]
        idx = torch.cat((pos_idx, neg_idx), dim=0)
        return pos_idx, neg_idx, idx

class FasterRCNN(nn.Module):
    def __init__(
        self,
        pre_nms_filter: int = 10_000,
        post_nms_topk: int = 2000,
        prop_filter_min_size: int = 16,
        prop_filter_score_thresh: float = 0.05,
        num_cls: int = 21,
        backbone_out_ch: int = 1024,
        hidden_dim: int = 1024,
        pool_size: Tuple[int, int] = (7, 7),
        box_nms_thresh: float = 0.5,
        detections_per_img: int = 100,
        box_score_thresh: float = 0.05
        ):
        super().__init__()
        self.rpn = RPN(
            pre_nms_filter=pre_nms_filter,
            post_nms_topk=post_nms_topk,
            prop_filter_min_size=prop_filter_min_size,
            prop_filter_score_thresh=prop_filter_score_thresh,
        )
        self.num_cls = num_cls
        self.pool_size = pool_size
        self.box_nms_thresh = box_nms_thresh
        self.detections_per_img = detections_per_img
        self.box_score_thresh = box_score_thresh
        self.fc = nn.Sequential(
            nn.Linear(backbone_out_ch * self.pool_size[0] * self.pool_size[1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        # background (0) + 20 classes
        self.cls = nn.Linear(hidden_dim, num_cls)
        # class-specific bbox regression
        # for each ROI: predict 4 deltas for each class
        self.bbox_reg = nn.Linear(hidden_dim, num_cls * 4)
        # for sanity
        self.class2index = {
            "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
            "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
            "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
            "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17,
            "sofa": 18, "train": 19, "tvmonitor": 20
            }

    def forward(
        self,
        x: torch.Tensor,
        target_bbox: Optional[torch.Tensor] = None,
        target_cls: Optional[torch.Tensor] = None
        ):
        _, _, img_h, img_w = x.shape
        if (not self.training) or (target_bbox is None) or (target_cls is None):
            rpn = self.rpn(x=x, target=None)
            # proposals: [M, 4], fmap: [1, C, H, W]
            # where M = h*w*a
            px: torch.Tensor = rpn['proposals'][0]
            fmap: torch.Tensor = rpn['feat_map']

            if px.numel() == 0:
                return {
                    "boxes": px.new_zeros((0, 4)),
                    "scores": px.new_zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long, device=px.device),
                }

            # ROI pool -> [M, C, 7, 7]
            px_roi_pool_feat = torchvision.ops.roi_pool(
                input=fmap,
                boxes=[px],
                output_size=self.pool_size,
                spatial_scale=1.0 / self.rpn.stride
            )
            # flatten -> [M, C*7*7]
            px_roi_pool_feat = torch.flatten(px_roi_pool_feat, start_dim=1)
            # fc -> [M, hidden_dim]
            fc = self.fc(px_roi_pool_feat)
            # cls_scores: [M, num_cls]
            cls_scores = self.cls(fc)
            # bbox_delta_pred: [M, num_cls, 4]
            bbox_delta_pred = self.bbox_reg(fc).view(-1, self.num_cls, 4)
            return self.postprocess_detections(
                cls_scores=cls_scores,
                bbox_delta_pred=bbox_delta_pred,
                proposals=px,
                img_h=img_h,
                img_w=img_w
            )

        # ! --------- training ----------
        rpn = self.rpn(x=x, target=target_bbox)
        # px [B, h*w*a, 4] -> [h*w*a, 4]
        px: torch.Tensor = rpn['proposals'][0]
        # fmap has shape [B, C, H, W] -> [1, C, H, W]
        fmap: torch.Tensor = rpn['feat_map']
        # append proposals since the model needs some intential perfect matches
        # early on the training to stablilize it - they will perfect match itself
        # px has shape [M, 4] and gt_bbox has shape [N, 4] -> [M+N, 4]
        pxg = torch.cat((px, target_bbox), dim=0)
        # px_gt has shape [M+N, 4] px_to_labels shape [M+N]
        px_gt, px_labels = self.prop_to_gtbox(pxg, target_bbox, target_cls)
        _, _, sample_idx = self.rpn.sample_anchors(px_labels, 128, 0.25)
        # pxg shape [128, 4]; px_labels shape [128]; px_gt shape [128, 4]
        pxg = pxg[sample_idx]
        px_labels = px_labels[sample_idx]
        px_gt = px_gt[sample_idx]
        # delta required to get from pxg (proposal) to gt_box(target) dx shape [128, 4]
        dx = encode_deltas(pxg, px_gt)
        # ROI pooling shape will be [128, C, 7, 7]
        pxg_roi_pool_feat = torchvision.ops.roi_pool(
            input=fmap,
            boxes=[pxg],
            output_size=self.pool_size,
            spatial_scale=1.0 / self.rpn.stride
            )
        # ROI pooling shape will be [128, C*7*7] flattened
        pxg_roi_pool_feat = torch.flatten(pxg_roi_pool_feat, start_dim=1)
        # pass to fully connected layer shape [128, hidden_dim]
        fc = self.fc(pxg_roi_pool_feat)
        # pass to cls to get classification probs for 21 classes
        # this will be of shape [128, 21]
        cls_scores = self.cls(fc)
        # pass to get the bbox delta (corrections/shifts) of shape [128, 21 * 4]
        bbox_delta_pred = self.bbox_reg(fc)
        # convert shape to get deltas for each cls of shape [128, 21, 4]
        bbox_delta_pred = bbox_delta_pred.view(-1, self.num_cls, 4)
        # roi_reg_loss only on positive proposals > 0
        # bbox_delta_pred[pos_idx] -> [P, num_cls, 4]
        # px_labels[pos_idx] are class ids in 1..C
        pos_idx = torch.where(px_labels > 0)[0]
        if pos_idx.numel() == 0:
            pred_pos = None
            target_pos = None
        else:
            # pred_pos -> [P, 4]
            pred_pos = bbox_delta_pred[pos_idx, px_labels[pos_idx]]
            target_pos = dx[pos_idx]

        return {
            "rpn_cls_input": rpn['cls_input'],
            "rpn_cls_target": rpn['cls_target'],
            "rpn_reg_input": rpn['reg_input'],
            "rpn_reg_target": rpn['reg_target'],
            "roi_cls_input": cls_scores,
            "roi_cls_target": px_labels,
            # roi_reg_loss only on positive proposals
            "roi_reg_input": pred_pos,
            "roi_reg_target": target_pos
        }

    def prop_to_gtbox(self, px, gt_bbox, gt_labels):
        # px (h*w*a + N, 4) here is the corrected-predictions/proposals given M=h*w*a
        # corrected after anchor+delta=proposal (apply_delta) format xyxy
        # removed B dim - has shape [M+N, 4]; gt_bbox shape [N, 4]; gt_labels shape [N]
        if gt_bbox.numel() == 0:
            px_gt = torch.zeros_like(px)
            labels = torch.zeros(px.size(0), dtype=torch.long, device=px.device)
            return px_gt, labels
        # this has shape [M + N, N] for each prop iou values per gt_bbox
        iou_mat = torchvision.ops.box_iou(px, gt_bbox)
        # px_gt_val has the max iou for a gt per anchor; px_gt_idx has the idx of the gt with max iou
        # both have shape [M + N]
        px_gt_val, px_gt_idx = torch.max(iou_mat, dim=-1)
        # we assign 0 for background (default) and class labels to foreground; labels shape [M + N]
        labels = torch.where(px_gt_val >= 0.5, gt_labels[px_gt_idx], 0).to(dtype=torch.long)
        # we now get gtbox for each proposal given its max_idx this has shape [M + N, 4]
        px_gt = gt_bbox[px_gt_idx]
        return px_gt, labels

    def clip_boxes_to_image(self, boxes, img_h, img_w):
        boxes = boxes.clone()
        boxes[:, 0] = boxes[:, 0].clamp(0, img_w)
        boxes[:, 1] = boxes[:, 1].clamp(0, img_h)
        boxes[:, 2] = boxes[:, 2].clamp(0, img_w)
        boxes[:, 3] = boxes[:, 3].clamp(0, img_h)
        return boxes

    @torch.inference_mode()
    def postprocess_detections(self, cls_scores, bbox_delta_pred, proposals, img_h, img_w):
        # cls_scores: [M, num_cls]
        # bbox_delta_pred: [M, num_cls, 4]
        # proposals: [M, 4]
        probs = nn.functional.softmax(cls_scores, dim=1)
        all_boxes, all_scores, all_labels = [], [], []
        # skip background class 0
        for c in range(1, self.num_cls):
            scores_c = probs[:, c] # [R]
            keep = scores_c >= self.box_score_thresh
            if keep.sum() == 0:
                continue

            props_c = proposals[keep] # [Rc, 4]
            deltas_c = bbox_delta_pred[keep, c] # [Rc, 4]
            scores_c = scores_c[keep] # [Rc]
            # decode proposal -> final class-specific box
            boxes_c = apply_deltas(props_c, deltas_c)  # [Rc, 4]
            boxes_c = self.clip_boxes_to_image(boxes_c, img_h, img_w)

            ws = boxes_c[:, 2] - boxes_c[:, 0]
            hs = boxes_c[:, 3] - boxes_c[:, 1]
            size_keep = (ws > 1) & (hs > 1)
            boxes_c = boxes_c[size_keep]
            scores_c = scores_c[size_keep]
            if boxes_c.numel() == 0:
                continue

            keep_idx = torchvision.ops.nms(boxes_c, scores_c, self.box_nms_thresh)
            boxes_c = boxes_c[keep_idx]
            scores_c = scores_c[keep_idx]
            labels_c = torch.full(
                (boxes_c.size(0),), c, dtype=torch.long, device=boxes_c.device
            )
            all_boxes.append(boxes_c)
            all_scores.append(scores_c)
            all_labels.append(labels_c)
        if len(all_boxes) == 0:
            return {
                "boxes": proposals.new_zeros((0, 4)),
                "scores": proposals.new_zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long, device=proposals.device),
            }
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        # keep top final detections
        if scores.numel() > self.detections_per_img:
            order = torch.argsort(scores, descending=True)[:self.detections_per_img]
            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order]
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
