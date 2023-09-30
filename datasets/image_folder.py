import os
import json
import torch
import pickle
import imageio
import numpy as np

from PIL import Image
from datasets import register
from torchvision import transforms
from torch.utils.data import Dataset

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, inp_size=None, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.inp_size = inp_size

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            img = transforms.ToTensor()(Image.open(x).convert('RGB'))
            if self.inp_size is None:
                return img

            img = transforms.Resize(self.inp_size, transforms.InterpolationMode.BILINEAR)(img)
            return img


        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('image-folder-eval')
class ImageFolderEval(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            img = transforms.ToTensor()(Image.open(x).convert('RGB'))
            return img

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('image-folder-resize')
class ImageFolder_resize(Dataset):
    def __init__(self, root_path, inp_size=(128,128), split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.inp_size = inp_size

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                img = transforms.ToTensor()(Image.open(file).convert('RGB'))
                if self.inp_size is not None:
                    img = transforms.Resize(self.inp_size, transforms.InterpolationMode.BILINEAR)(img)
                self.files.append(img)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            img = transforms.ToTensor()(Image.open(x).convert('RGB'))
            if self.inp_size is not None:
                img = transforms.Resize(self.inp_size, transforms.InterpolationMode.BILINEAR)(img)
            return img

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('image-folder-resize-np')
class ImageFolder_resize(Dataset):
    def __init__(self, root_path, inp_size=(128,128), split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.inp_size = inp_size

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                img = transforms.ToTensor()(Image.open(file).convert('RGB'))
                img = transforms.Resize(self.inp_size, transforms.InterpolationMode.BILINEAR)(img)
                self.files.append(img.numpy())

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            img = transforms.ToTensor()(Image.open(x).convert('RGB'))
            if self.inp_size is not None:
                img = transforms.Resize(self.inp_size, transforms.InterpolationMode.BILINEAR)(img)
            return img.permute(1, 2, 0).numpy()

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('numpy-folder')
class NumpyFolder(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            self.files.append(np.load(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return x

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]


@register('stit-tgt-mask')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        self.dataset_3 = ImageFolder(root_path_3, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx],  self.dataset_3[idx]


@register('eval-align')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, root_path_4=None, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        self.dataset_3 = ImageFolder(root_path_3, **kwargs)
        self.dataset_4 = ImageFolder(root_path_4, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx],  self.dataset_3[idx], self.dataset_4[idx]


@register('eval-stitching')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, **kwargs):
        self.dataset_1 = ImageFolderEval(root_path_1, **kwargs)
        self.dataset_2 = ImageFolderEval(root_path_2, **kwargs)
        self.dataset_3 = ImageFolderEval(root_path_3, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx],  self.dataset_3[idx]


@register('eval-stitched-images')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, root_path_4, root_path_5, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        self.dataset_3 = ImageFolder(root_path_3, **kwargs)
        self.dataset_4 = ImageFolder(root_path_4, **kwargs)
        self.dataset_5 = ImageFolder(root_path_5, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx], self.dataset_4[idx], self.dataset_5[idx]


@register('eval-shift')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, inp_size, **kwargs):
        self.dataset_1 = ImageFolder_resize(root_path_1, inp_size, **kwargs)
        self.dataset_2 = ImageFolder_resize(root_path_2, inp_size, **kwargs)
        self.dataset_3 = NumpyFolder(root_path_3, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx],  self.dataset_3[idx]


@register('udis')
class PairedImageTransformFolders(Dataset):
    def __init__(self, root_path_1, root_path_2, root_path_3, root_path_4, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        self.dataset_3 = ImageFolder(root_path_3, **kwargs)
        self.dataset_4 = ImageFolder(root_path_4, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx], self.dataset_4[idx]


@register('udis-sweep')
class PairedImageTransformFolders(Dataset):
    def __init__(self, root_path_1, root_path_2, root_path_3, root_path_4, inp_size, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, inp_size, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, inp_size, **kwargs)
        self.dataset_3 = ImageFolder(root_path_3, inp_size, **kwargs)
        self.dataset_4 = ImageFolder(root_path_4, inp_size, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx], self.dataset_4[idx]


@register('paired-image-folders-resize')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, inp_size=None, **kwargs):
        self.dataset_1 = ImageFolder_resize(root_path_1, inp_size, **kwargs)
        self.dataset_2 = ImageFolder_resize(root_path_2, inp_size, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]


@register('paired-image-folders-with-shift')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        self.dataset_3 = NumpyFolder(root_path_3, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx]


@register('paired-image-with-mask-and-grid-folders')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, root_path_4, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        self.dataset_3 = ImageFolder(root_path_3, **kwargs)
        self.dataset_4 = NumpyFolder(root_path_4, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx], self.dataset_4[idx]


@register('paired-images-stit-H-size')
class PairedImageTransformFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, root_path_4, root_path_5, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        self.dataset_3 = ImageFolder(root_path_3, **kwargs)
        self.dataset_4 = NumpyFolder(root_path_4, **kwargs)
        self.dataset_5 = NumpyFolder(root_path_5, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx], self.dataset_4[idx], self.dataset_5[idx]
