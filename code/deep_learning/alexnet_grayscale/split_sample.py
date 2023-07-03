import os
from shutil import copy, rmtree
import random
random.seed(0)

def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


def clear_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
img_size = 32
split_rate = 0.2

data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sample_path = os.path.join(data_root,'data','img2_gray_size'+str(img_size))
origin_path = os.path.join(sample_path,'sample')

assert os.path.exists(origin_path), "path '{}' does not exist.".format(origin_path)

classes = [cla for cla in os.listdir(origin_path)
           if os.path.isdir(os.path.join(origin_path, cla))]


train_root = os.path.join(sample_path, "train")
mk_file(train_root)
clear_dir(train_root)
for cla in classes:
    mk_file(os.path.join(train_root, cla))
    

val_root = os.path.join(sample_path, "val")
mk_file(val_root)
clear_dir(val_root)
for cla in classes:
    mk_file(os.path.join(val_root, cla))

for cla in classes:
    cla_path = os.path.join(origin_path, cla)
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join(val_root, cla)
            copy(image_path, new_path)
        else:
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join(train_root, cla)
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")
    print()

print("processing done!")


