from utils import CONFIG
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.amg import build_all_layer_point_grids
import torch
from torch import nn


def get_sam(return_generator=True):
    sam = sam_model_registry["default"](checkpoint=CONFIG["model-path"])
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


if __name__ == "__main__":
    sam = get_sam(return_generator=False)
    simp_sam = SimpleSAM(sam)
    sparse_prompt_embedding, dense_prompt_embedding = (
        simp_sam.sparse_prompt_emb,
        simp_sam.dense_prompt_emb,
    )
    print(sparse_prompt_embedding.shape, dense_prompt_embedding.shape)
