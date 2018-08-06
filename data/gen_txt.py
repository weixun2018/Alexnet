import os

dataset_dir = 'G:\\competition\\train'
print(os.listdir(dataset_dir))
abs_dir = os.path.join(os.getcwd(), dataset_dir)
directory = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(abs_dir, d))]

direc = [d for d in directory if len(os.listdir(os.path.join(dataset_dir, d))) > 0]
print(direc)

label_class_mapping = {}
for i, d in enumerate(sorted(direc)):
    label_class_mapping[d] = i

print(label_class_mapping)

train_image_label_list = []
val_image_label_list = []
train_fraction = 0.9
for d in direc:
    dr = os.path.join(abs_dir, d)
    cls = label_class_mapping[d]
    length = len(os.listdir(dr))
    for i, img_name in enumerate(os.listdir(dr)):
        image_label_str = str(os.path.join(dr, img_name)) + ' ' + str(cls)
        if i < length * train_fraction:
            train_image_label_list.append(image_label_str)
        else:
            val_image_label_list.append(image_label_str)

print(train_image_label_list[:10])
print(len(train_image_label_list), len(val_image_label_list))

train_file = 'train.txt'
val_file = 'val.txt'

with open(train_file, 'w') as tf:
    for line in train_image_label_list:
        tf.write(line + '\n')

with open(val_file, 'w') as vf:
    for line in val_image_label_list:
        vf.write(line + '\n')
