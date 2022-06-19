import torch
from PIL import Image
from torch.utils.data import Dataset
from easydl.common.wheel import *
import numpy as np

class BaseImageDataset(Dataset):
    """
    base image dataset

    for image dataset, ``__getitem__`` usually reads an image from a given file path

    the image is guaranteed to be in **RGB** mode

    subclasses should fill ``datas`` and ``labels`` as they need.
    """

    def __init__(self, transform=None, return_id=False):
        self.return_id = return_id
        self.transform = transform or (lambda x : x)
        self.datas = []
        self.labels = []
        self.test_datas = []
        self.test_labels = []

    def __getitem__(self, index):
        im = Image.open(self.datas[index]).convert('RGB')
        im = self.transform(im)
        if not self.return_id:
            return im, self.labels[index]
        return im, self.labels[index], index

    def __len__(self):
        return len(self.datas)


class FileListDataset_splittrain(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :

    image_path label_id
    image_path label_id
    ......

    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=True, num_classes=None, filter=None, num_per_class=10):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(FileListDataset_splittrain, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines() if
                    line.strip()]  # avoid empty lines
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        classes = np.unique(self.labels)
        count = np.zeros((len(classes), ))
        i = 0
        while(i < len(self.datas)):
            count[self.labels[i]] += 1
            if count[self.labels[i]] > num_per_class:
                self.datas.pop(i)
                self.labels.pop(i)
                i = i - 1
            i = i + 1

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)

        self.num_classes = num_classes or max(self.labels) + 1

class FileListDataset_splittest(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :

    image_path label_id
    image_path label_id
    ......

    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=True, num_classes=None, filter=None, num_per_class=10):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(FileListDataset_splittest, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines() if
                    line.strip()]  # avoid empty lines
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        classes = np.unique(self.labels)
        count = np.zeros((len(classes), ))
        i = 0
        while(i < len(self.datas)):
            if count[self.labels[i]] < num_per_class:
                count[self.labels[i]] += 1
                self.datas.pop(i)
                self.labels.pop(i)
                i = i - 1
            i = i + 1
        i = 0
        count = np.zeros((len(classes), ))
        while(i < len(self.datas)):
            count[self.labels[i]] += 1
            if count[self.labels[i]] > num_per_class:
                self.datas.pop(i)
                self.labels.pop(i)
                i = i - 1
            i = i + 1

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)

        self.num_classes = num_classes or max(self.labels) + 1


        

class FileListDataset_unlabeled(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :

    image_path label_id
    image_path label_id
    ......

    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None, num_per_class=10):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(FileListDataset_unlabeled, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines() if
                    line.strip()]  # avoid empty lines
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e
                
        classes = np.unique(self.labels)
        count = np.zeros((len(classes), ))
        i = 0
        while(i < len(self.datas)):
            count[self.labels[i]] += 1
            if count[self.labels[i]] > num_per_class:
                self.datas.pop(i)
                self.labels.pop(i)
                i = i - 1
            i = i + 1

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)

        self.num_classes = num_classes or max(self.labels) + 1