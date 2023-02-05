import torch
from torch.utils.data import Dataset

def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        yield large_batch, model_kwargs

class ConditionedImagesDataset(Dataset):
    def __init__(self, hi_res_images: torch.Tensor, low_res_images: torch.Tensor = None):
        self.hi_res_images = hi_res_images
        
        if low_res_images is None:
            # random or center crop