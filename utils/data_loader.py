import os
import nibabel as nib
from glob import glob

from torch.utils.data import Dataset

class FeaturePointsDataset(Dataset):
    
    def __init__(self,img_dir:str,transform = None,) -> None:
        images = sorted(glob(os.path.join(img_dir, "img*.nii.gz")))
        labels = sorted(glob(os.path.join(img_dir, "label*.nii.gz")))
        self.files = [{"img": img, "label": label} for img, label in zip(images, labels)]
        self.transform = transform

    def __len__(self):#返回数据集的大小
        return len(self.files)
    
    def __getitem__(self, index):
        imgs = self.files[index]
        img = nib.load(imgs["img"])
        label = nib.load(imgs["label"])
        img_data ={'img':img,'label':label}
        
        if self.transform:
            img_data = self.transform(img_data)

        return img_data


if __name__ == "__main__":
    myDataset = FeaturePointsDataset('/home/xindong/project/data/x-ray')

    img_data = myDataset.__getitem__(1)
    print(f"len:{len(myDataset)}")
    print(img_data['img'].shape)
