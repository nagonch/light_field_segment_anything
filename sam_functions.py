from utils import CONFIG
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.amg import build_all_layer_point_grids
import torch
from torch import nn
from torchvision.transforms.functional import resize


def get_sam(return_generator=True):
    sam = sam_model_registry["vit_b"](checkpoint=CONFIG["model-path"])
    sam = sam.to(device="cuda")
    if not return_generator:
        return sam
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


class SimpleSAM(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.sparse_prompt_emb, self.dense_prompt_emb = self.get_prompt_embeddings()

    @torch.no_grad()
    def get_prompt_embeddings(self, n_points_per_side=32):
        input_points = torch.tensor(
            build_all_layer_point_grids(n_points_per_side, 0, 1)[0]
        ).cuda()
        input_labels = torch.tensor([1 for _ in range(input_points.shape[0])]).cuda()[
            None
        ]
        input_points = input_points[None]
        sparse_prompt_embedding, dense_prompt_embedding = self.sam.prompt_encoder(
            (input_points, input_labels), None, None
        )
        return sparse_prompt_embedding, dense_prompt_embedding

    @torch.no_grad()
    def preprocess_batch(self, batch):
        batch = resize(
            batch.reshape(-1, batch.shape[-2], batch.shape[-1]),
            (1024, 1024),
            antialias=True,
        ).reshape(batch.shape[0], 3, 1024, 1024)
        batch = torch.stack([self.sam.preprocess(x) for x in batch], dim=0)
        return batch

    @torch.no_grad()
    def forward(
        self,
        batch,
    ):
        img_original_size = batch.shape[-2:]
        batch = self.preprocess_batch(batch)
        image_embeddings = self.sam.image_encoder(batch)
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.sparse_prompt_emb,
            dense_prompt_embeddings=self.dense_prompt_emb,
            multimask_output=True,
        )
        masks = self.sam.postprocess_masks(
            low_res_masks,
            input_size=batch.shape[-2:],
            original_size=img_original_size,
        )
        masks = masks > self.sam.mask_threshold
        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }


if __name__ == "__main__":
    import os
    from PIL import Image
    import numpy as np
    from segment_anything import SamPredictor
    from torchvision.transforms.functional import resize
    from matplotlib import pyplot as plt
    import imgviz

    sam = get_sam(return_generator=False)
    simple_sam = SimpleSAM(sam)
    dir = "/home/cedaradmin/data/lf_angular/LFPlane/f00051/png"
    subviews = []
    for img in list(sorted(os.listdir(dir))):
        path = dir + "/" + img
        subviews.append(np.array(Image.open(path))[:, :, :3])
    LF = np.stack(subviews).reshape(17, 17, 128, 128, 3).astype(np.uint8)
    batch = (torch.as_tensor(LF[0:2, 0]).permute(0, -1, 1, 2).float()).cuda()
    result = simple_sam(batch)
    masks = result["masks"].to(torch.int32)
    img1 = batch[1].permute(1, 2, 0).detach().cpu().numpy()
    masks1 = masks[1].detach().cpu().numpy()
    for mask in masks1:
        plt.imshow(mask, cmap="gray")
        # vis = imgviz.label2rgb(
        #     label=mask,
        #     image=img1 * (mask == 1)[:, :, None],
        #     colormap=imgviz.label_colormap(mask.max() + 1),
        # )
        # plt.imshow(vis)
        plt.show()
