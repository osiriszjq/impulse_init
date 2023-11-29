import torch
import os

# my srank
def srank_l12(X):
    (u,s,v) = torch.svd(X)
    sr2 = (s*s).sum()/s[0]/s[0]
    sr1 = s.sum()/s[0]
    return sr1,sr2


# my counting parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_val_img_folder(args):
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    dataset_dir = os.path.join('/data', 'tiny-imagenet-200')
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def get_class_name(args):
    class_to_name = dict()
    fp = open(os.path.join(args.data_dir, args.dataset, 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name[words[0]] = words[1].split(',')[0]
    fp.close()
    return class_to_name