import os
import shutil
import xml.dom.minidom
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--root', type=str, default='/eva_data0/imagenet')
parser.add_argument('--val-img-dir', type=str, default='img_val_raw')
parser.add_argument('--val-annot-dir', type=str, default='bbox_val_v3/val')
parser.add_argument('--train-img-dir', type=str, default='img_train')
parser.add_argument('--save-dir', type=str, default='img_val')

args = parser.parse_args()

val_img_dir = os.path.join(args.root, args.val_img_dir)
val_annot_dir = os.path.join(args.root, args.val_annot_dir)
train_img_dir = os.path.join(args.root, args.train_img_dir)
save_dir = os.path.join(args.root, args.save_dir)

if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    class_dirs = os.listdir(train_img_dir)
    for class_dir in class_dirs:
        class_path = os.path.join(save_dir, class_dir)
        os.mkdir(class_path)

for image_name in os.listdir(val_img_dir):
    name = image_name.split('.')[0]
    annot_path = os.path.join(val_annot_dir, name)+'.xml'
    dom = xml.dom.minidom.parse(annot_path)
    name_attr = dom.getElementsByTagName("name")[0]
    img_dir = name_attr.firstChild.data
    src_path = os.path.join(val_img_dir, image_name)
    dst_path = os.path.join(save_dir, img_dir, image_name)
    shutil.copy(src_path, dst_path)
    print(f"{src_path} -> {dst_path}")