from torch.utils import data

class CustomWrapper(data.Dataset):
    def __init__(self, original_dataset, augmentations):
        self.original_dataset = original_dataset
        self.augmentations = augmentations

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # original should output img with type PIL Image or Tensor
        img, label = self.original_dataset[idx]
        return self.augmentations(img), label