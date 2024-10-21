from custom_dataset import CustomDataset
from torch.utils.data import DataLoader, random_split

def get_dataloaders(filename, annFile, transform, batch_size, train_size, get_key_conn=True):
    dataset = CustomDataset(filename, annFile, transform)
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    if get_key_conn:
        return train_dataloader, test_dataloader, dataset.get_keypoints(), dataset.get_connections()
    else:
        return train_dataloader, test_dataloader
