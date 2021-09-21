import os
from glob import glob

from torch.utils.data import Dataset

class FeaturePointsDataset(Dataset):
    
    def __init__(self,img_dir:str,transform = None,) -> None:
        super().__init__()
        images = sorted(glob(os.path.join(img_dir, "img*.nii.gz")))
        labels = sorted(glob(os.path.join(img_dir, "label*.nii.gz")))
        self.files = [{"img": img, "label": label} for img, label in zip(images, labels)]

        self.img_dir