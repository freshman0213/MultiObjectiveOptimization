import torch
from torchvision import transforms
from loaders.multi_mnist_loader import MNIST
from loaders.cityscapes_loader import CITYSCAPES
from loaders.segmentation_augmentations import *
from loaders.celeba_loader import CELEBA
from torchvision.datasets import CIFAR10, SVHN

from loaders.custom_wrapper import CustomWrapper

# Setup Augmentations
cityscapes_augmentations= Compose([RandomRotate(10),
                                   RandomHorizontallyFlip()])

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(params, configs):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'mnist' in params['dataset'] or 'mnist_film' in params['dataset']:
        train_dst = MNIST(root=configs['mnist']['path'], split='train', download=True, transform=global_transformer())
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)

        val_dst = MNIST(root=configs['mnist']['path'], split='val', download=True, transform=global_transformer())
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=False, num_workers=4)

        test_dst = MNIST(root=configs['mnist']['path'], split='test', download=True, transform=global_transformer())
        test_loader = torch.utils.data.DataLoader(test_dst, batch_size=100, shuffle=False, num_workers=4)
        return train_loader, train_dst, val_loader, val_dst, test_loader, test_dst

    if 'cityscapes' in params['dataset']:
        train_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['train'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']), augmentations=cityscapes_augmentations)
        val_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['val'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']))

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
        return train_loader, train_dst, val_loader, val_dst

    if 'celeba' in params['dataset']:
        #TODO
        train_dst = CELEBA(params, root=configs['celeba']['path'], is_transform=True, split='train', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
        val_dst = CELEBA(params, root=configs['celeba']['path'], is_transform=True, split='val', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
        #TODO
        return train_loader, train_dst, val_loader, val_dst, val_loader, val_dst

    if 'cifar_svhn' in params['dataset']: 
        
        train_val_dst = CIFAR10(root='./PATH_FOR_CIFAR10_DATASET', train=True, download=True)
        train_size = int(len(train_val)*0.8)
        val_size = len(train_val) - train_size
        print('CIFAR10 - train:', train_size)
        print('CIFAR10 - val:', val_size)
        cifar_train_val = torch.utils.data.random_split(train_val_dst, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        cifar_augmentation = transforms.Compose(
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(), 
        )
        cifar_normalization = transforms.Compose(
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
        )
        
        cifar_train_dst = CustomWrapper(cifar_train_val[0], transforms.Compose([cifar_augmentation, cifar_normalization]))
        cifar_train_loader = torch.utils.data.DataLoader(cifar_train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4, drop_last=True)
        cifar_val_dst = CustomWrapper(cifar_train_dst[1], transforms.Compose([cifar_augmentation, cifar_normalization]))
        cifar_val_loader = torch.utils.data.DataLoader(cifar_val_dst, batch_size=params['batch_size'], shuffle=False, num_workers=4, drop_last=True)

        cifar_test_dst = CIFAR10(root='./PATH_FOR_CIFAR10_DATASET', train=False, download=True, transform=cifar_normalization)
        cifar_test_loader = torch.utils.data.DataLoader(cifar_test_dst, batch_size=params['batch_size'], shuffle=False, num_workers=4)

        
        train_val = SVHN(root='./PATH_FOR_SVHN_DATASET', split='train', download=True)
        train_size = int(len(train_val)*0.8)
        val_size = len(train_val) - train_size
        print('SVHN - train:', train_size)
        print('SVHN - val:', val_size)
        svhn_train_val = torch.utils.data.random_split(train_val, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        svhn_augmentation = transforms.Compose(
            transforms.RandomRotation(8),
            transforms.RandomResizedCrop(32, (0.95, 1.05), (1, 1)),
            transforms.RandomAffine(0, shear=0.15)
        )
        svhn_normalization = transforms.Compose(
            transforms.ToTensor(),
            transforms.Normalize()
        )
        svhn_train_loader = torch.utils.data.DataLoader(svhn_train_val[0], batch_size=params['batch_size'], shuffle=True, num_workers=4, drop_last=True)
        svhn_val_loader = torch.utils.data.DataLoader(svhn_train_val[1], batch_size=params['batch_size'], shuffle=False, num_workers=4, drop_last=True)

        svhn_test_dst = SVHN(root='./PATH_FOR_SVHN_DATASET', split='test', download=True, transform=transform)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_test_dst, batch_size=params['batch_size'], shuffle=False, num_workers=4)


        # for batch in cifar_train_loader:
        #     print(batch[0][0], batch[1][0])
        #     break

        # for batch in svhn_train_loader:
        #     print(batch[0][0], batch[1][0])
        #     break

        return (cifar_train_loader, svhn_train_loader), (cifar_train_val[0], svhn_train_val[0]), \
               (cifar_val_loader, svhn_val_loader), (cifar_train_val[1], svhn_train_val[1]), \
               (cifar_test_loader, svhn_test_loader), (cifar_test_dst, svhn_test_dst)