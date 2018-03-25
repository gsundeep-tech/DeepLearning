def download_dataset():
    
    urls = {
        'train_images' : 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels' : 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images' : 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels' : 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
    
    from urllib.request import urlretrieve
    from os.path import isfile, isdir
    from tqdm import tqdm
    import tarfile


    mnist_dataset_folder_path = 'data'

    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num
    
    for key, value in urls.items():
        file_path = value.split('/')[-1]
        if not isfile(file_path):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc=key) as pbar:
                urlretrieve(
                    value,
                    file_path,
                    pbar.hook)

                
def load_data():
    
    from tensorflow.examples.tutorials.mnist import input_data
    
    print('Getting MNIST Dataset...')
    mnist = input_data.read_data_sets("./", one_hot=True)
    print('Data Extracted.')
    
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    
    validation_images = mnist.validation.images
    validation_labels = mnist.validation.labels
    
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    
    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels

def reshape(x):
    
    import numpy as np
    
    return np.reshape(x, [-1, 28, 28, 1])