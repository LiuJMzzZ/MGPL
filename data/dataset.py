from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import subprocess
import os

data_dir = './data'

class TinyImagenetDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=True):
        super(TinyImagenetDataset, self).__init__()
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self._download()
        self.labels_list = self._retrieve_labels_list()
        self.image_paths, self.labels = self._get_data()

    def _download(self):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        if not os.path.exists(f'{self.data_dir}/cs231n.stanford.edu/tiny-imagenet-200.zip'):
            subprocess.run(f'wget -r -nc -P {self.data_dir} {url}'.split())
            subprocess.run(f'unzip -qq -n {self.data_dir}/cs231n.stanford.edu/tiny-imagenet-200.zip -d {self.data_dir}'.split())

    def _retrieve_labels_list(self):
        labels_list = []
        with open(f'{self.data_dir}/tiny-imagenet-200/wnids.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    labels_list += [line]
        return labels_list

    def _get_data(self):
        image_paths, labels = [], []

        # If train
        if self.train:
            for cl_folder in sorted(os.listdir(f'{self.data_dir}/tiny-imagenet-200/train')):
                label = self.labels_list.index(cl_folder)
                for image_name in sorted(os.listdir(f'{self.data_dir}/tiny-imagenet-200/train/{cl_folder}/images')):
                    image_path = f'{self.data_dir}/tiny-imagenet-200/train/{cl_folder}/images/{image_name}'
                    image_paths += [image_path]
                    labels += [label]

        # If validation
        else:
            with open(f'{self.data_dir}/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    image_name, label_str = line.split('\t')[:2]
                    image_path = f'{self.data_dir}/tiny-imagenet-200/val/images/{image_name}'
                    label = self.labels_list.index(label_str)
                    image_paths += [image_path]
                    labels += [label]
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


class dataset_transform(Dataset):
    def __init__(self, dataset, select_classes=None, target_transform=None):
        self.dataset = dataset
        self.target_transform = target_transform
        if select_classes is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = [idx for idx, (_, y) in enumerate(dataset) if y in select_classes]
            self.transform_dict = dict(zip(select_classes, list(range(len(select_classes)))))
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.dataset[self.indices[idx]][1]
        if self.target_transform=='reindex':
            label = self.transform_dict[label]
        elif self.target_transform=='open':
            label = 999
        else:
            label = label
        return (image, label)
    def __len__(self):
        return len(self.indices)


def get_dataset(dataset, train=False, select_classes=None, target_transform=None):
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ])
        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                ])
        train_set = datasets.CIFAR10(root=data_dir,train=True,transform=transform_train,download=True)
        test_set = datasets.CIFAR10(root=data_dir,train=False,transform=transform_test,download=True)

    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ])
        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                ])
        train_set = datasets.CIFAR100(root=data_dir,train=True,transform=transform_train,download=True)
        test_set = datasets.CIFAR100(root=data_dir,train=False,transform=transform_test,download=True)

    elif dataset == 'svhn':
        transform_train = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                ])
        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                ])
        train_set = datasets.SVHN(root=data_dir,split='train',transform=transform_train, download=True)
        test_set = datasets.SVHN(root=data_dir,split='test',transform=transform_test, download=True)
    
    elif dataset == 'tiny_imagenet':
        transform_train = transforms.Compose([
                                transforms.Resize(64),
                                transforms.RandomRotation(20),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
                                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                                ])
        transform_test = transforms.Compose([
                                transforms.Resize(64),
                                transforms.ToTensor(),
                                lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
                                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                                ])
        train_set = TinyImagenetDataset(data_dir=data_dir,train=True,transform=transform_train, download=True)
        test_set = TinyImagenetDataset(data_dir=data_dir,train=False,transform=transform_test, download=True)

    if train:
        return dataset_transform(train_set, select_classes, target_transform)
    else:
        return dataset_transform(test_set, select_classes, target_transform)



if __name__ == '__main__':
    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]
    total_classes = list(range(10))
    unknown_classes = splits[0]
    known_classes = list(set(total_classes) - set(unknown_classes))

    train_set = get_dataset('cifar10', True, known_classes, 'reindex')
    test_set = get_dataset('cifar10', False, known_classes, 'reindex')
    open_set = get_dataset('cifar10', False, unknown_classes, 'open')    

