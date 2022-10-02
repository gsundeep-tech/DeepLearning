#importing libraries
import numpy as np

# downloading MNIST datasets
def download_dataset():
    
    from urllib.request import urlretrieve
    from os.path import isfile, isdir
    from tqdm import tqdm
    import tarfile


    cifar10_dataset_folder_path = 'cifar-10-batches-py'
    tar_gz_path = 'cifar-10-python.tar.gz'

    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    if not isfile(tar_gz_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                tar_gz_path,
                pbar.hook)

    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open(tar_gz_path) as tar:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(tar)
            tar.close()

def one_hot_encoding(x):
    
    n_values = 10
    return np.eye(n_values)[x]

def normalize(x):
    
    return x/255

def load_images_labels():
    
    import pickle
    from sklearn.utils import shuffle
    
    batch_ids = 5
    train_validation_dataset = []
    
    #preprocess the images dataset
    cifar10_dataset_folder_path = 'cifar-10-batches-py'
    
    for batch_id in range(1, batch_ids + 1):
        with open(cifar10_dataset_folder_path + "/data_batch_" + str(batch_id), mode='rb' ) as file:
            batch = pickle.load(file, encoding='latin1')
            for data, label in zip(batch['data'], batch['labels']):
                train_validation_dataset.append((normalize(data), one_hot_encoding(label)))
    
    test_dataset = []
    with open(cifar10_dataset_folder_path + "/test_batch", mode='rb') as file:
        for data, label in zip(batch['data'], batch['labels']):
            test_dataset.append((normalize(data), one_hot_encoding(label)))
            
    return shuffle(shuffle(train_validation_dataset)), test_dataset

def extract_images_labels(images_labels):
    
    images = np.array([x[0] for x in images_labels])
    labels = np.array([x[1] for x in images_labels])
    
    return images, labels