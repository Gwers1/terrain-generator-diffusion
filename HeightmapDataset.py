from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

class HeightmapDataset(Dataset):
    def __init__(self, dir, transform = None):
        self.dir = dir
        self.transform = transform
        self.images = datasets.ImageFolder(dir).imgs

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx][0]
        image = Image.open(image_path)

        if self.transform:
            for x in self.transform:
                image = x(image)
        return image
    
# heightMapDatasetTest = HeightmapDataset(dir= 'resources/')
# for i, image in enumerate(heightMapDatasetTest):
#     print(i, image)