import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SignatureDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        
        self.real_images = [(os.path.join(self.real_dir, str(i), img), 1) for i in range(402) if os.path.exists(os.path.join(self.real_dir, str(i))) for img in os.listdir(os.path.join(self.real_dir, str(i)))]
        self.fake_images = [(os.path.join(self.fake_dir, str(i), img), 0) for i in range(402) if os.path.exists(os.path.join(self.fake_dir, str(i))) for img in os.listdir(os.path.join(self.fake_dir, str(i)))]
        self.total_images = self.real_images + self.fake_images

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        image_path, label = self.total_images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 主函数
if __name__ == '__main__':

    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet standards
])

    batch_size = 32
    path = "F:\\LMX\\QianMing\\Data\\SigData_1"
    real_dir = os.path.join(path, "real_signatures")
    fake_dir = os.path.join(path, "fake_signatures")
    dataset = SignatureDataset(real_dir=real_dir, fake_dir=fake_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i, (images, labels) in enumerate(dataloader):
        print(images.shape)
        print(labels.shape)
        break