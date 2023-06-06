# from __future__ import print_function
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import torchvision.datasets as datasets
import numpy as np

class IncrementalCIFAR10(datasets.CIFAR10):

    def build_incremental_data_list(self, step_size=2):

        num_class = 10
        assert num_class % step_size == 0

        self.incremental_data_list = []
        self.task_id = -1
        self.targets_ = np.array(self.targets)
        for i in range(0, num_class, step_size):
            idx = None
            for j in range(step_size):
                IncrementalClass = i + j
                if idx is None:
                    idx = (self.targets_ == IncrementalClass)
                else:
                    idx = idx | (self.targets_ == IncrementalClass)
            self.incremental_data_list.append((self.data[idx], self.targets_[idx]))

    def build_accumulate_incremental_data_list(self, step_size=2):
        num_class = 10
        assert num_class % step_size == 0

        self.incremental_data_list = []
        self.task_id = -1
        self.targets_ = np.array(self.targets)
        for i in range(0, num_class, step_size):
            idx = None
            for j in range(i + step_size):
                IncrementalClass = j
                if idx is None:
                    idx = (self.targets_ == IncrementalClass)
                else:
                    idx = idx | (self.targets_ == IncrementalClass)
            self.incremental_data_list.append((self.data[idx], self.targets_[idx]))

    def reset_task(self, classid):
        self.data, self.targets = self.incremental_data_list[classid]
        self.task_id = classid

class IncrementalMNIST(datasets.MNIST):

    def build_incremental_data_list(self, step_size=2):

        num_class = 10
        assert num_class % step_size == 0

        self.incremental_data_list = []
        self.task_id = -1
        self.targets_ = np.array(self.targets)
        for i in range(0, num_class, step_size):
            idx = None
            for j in range(step_size):
                IncrementalClass = i + j
                if idx is None:
                    idx = (self.targets_ == IncrementalClass)
                else:
                    idx = idx | (self.targets_ == IncrementalClass)
            self.incremental_data_list.append((self.data[idx], self.targets_[idx]))

    def build_accumulate_incremental_data_list(self, step_size=2):
        num_class = 10
        assert num_class % step_size == 0

        self.incremental_data_list = []
        self.task_id = -1
        self.targets_ = np.array(self.targets)
        for i in range(0, num_class, step_size):
            idx = None
            for j in range(i + step_size):
                IncrementalClass = j
                if idx is None:
                    idx = (self.targets_ == IncrementalClass)
                else:
                    idx = idx | (self.targets_ == IncrementalClass)
            self.incremental_data_list.append((self.data[idx], self.targets_[idx]))

    def reset_task(self, classid):
        self.data, self.targets = self.incremental_data_list[classid]
        self.task_id = classid


def get_dataloader(data_name, root, batch_size, num_workers, class_per_step):

    data_name = data_name.lower()
    if data_name == "cifar10":

        transform_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        dataset_train = datasets.CIFAR10(root=root+'/cifar10', train=True, transform=transform_train, download=True)
        dataset_test = datasets.CIFAR10(root=root + '/cifar10', train=False, transform=transform_test, download=True)

        dataset_train_inc = IncrementalCIFAR10(root=root + '/cifar10', train=True, transform=transform_train, download=True)

        dataset_train_accinc = IncrementalCIFAR10(root=root + '/cifar10', train=True, transform=transform_train, download=True)
        dataset_test_accinc = IncrementalCIFAR10(root=root + '/cifar10', train=False, transform=transform_test, download=True)

        dataset_train_inc.build_incremental_data_list(step_size=class_per_step)

        dataset_train_accinc.build_accumulate_incremental_data_list(step_size=class_per_step)
        dataset_test_accinc.build_accumulate_incremental_data_list(step_size=class_per_step)

    elif data_name == 'mnist':
        transform_train = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])

        transform_test = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])
        dataset_train = datasets.MNIST(
            root=root+'/mnist', train=True, transform=transform_train, download=True)
        dataset_test = datasets.MNIST(
            root=root + '/mnist', train=False, transform=transform_test, download=True)

        dataset_train_inc = IncrementalMNIST(
            root=root + '/mnist', train=True, transform=transform_train, download=True)

        dataset_train_accinc = IncrementalMNIST(
            root=root + '/mnist', train=True, transform=transform_train, download=True)
        dataset_test_accinc = IncrementalMNIST(
            root=root + '/mnist', train=False, transform=transform_test, download=True)

        dataset_train_inc.build_incremental_data_list(step_size=class_per_step)
        dataset_train_accinc.build_accumulate_incremental_data_list(step_size=class_per_step)
        dataset_test_accinc.build_accumulate_incremental_data_list(step_size=class_per_step)
    else:
        raise ValueError()

    print("incremental data list size: ", len(dataset_train_accinc.incremental_data_list))
    print("++++++++++++++++++")
    for data, target in dataset_train_accinc.incremental_data_list:
        print("sub data size: ", data.shape)
    print("++++++++++++++++++")
        
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloader_train_inc = torch.utils.data.DataLoader(
        dataset_train_inc, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloader_train_accinc = torch.utils.data.DataLoader(
        dataset_train_accinc, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test_accinc = torch.utils.data.DataLoader(
        dataset_test_accinc, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return (dataloader_train, dataloader_test), dataloader_train_inc, (dataloader_train_accinc, dataloader_test_accinc)
