import cv2
import tensorflow as tf
from alexnet import AlexNet
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_label_file = 'data/train.txt'
checkpoints_path = 'result/checkpoints'
test_image_limit = 20

path_list = []
label_list = []
label_class_mapping = {'drawing': 0, 'graffiti': 1, 'watercolor': 2, 'vector': 3}
class_label_mapping = {v: k for k, v in label_class_mapping.items()}
print(class_label_mapping)
with open(image_label_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        path_label = line.split(' ')
        path_list.append(path_label[0])
        label_list.append(int(path_label[1]))

print(len(path_list), len(set(path_list)))

rand = np.random.randint(0, len(lines)-1, test_image_limit)
path_list = np.array(path_list)[rand]
label_list = np.array(label_list)[rand]
# print(label_list)

image_list = [cv2.imread(p) for p in path_list]

# print(len(label_list), label_list)

x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)
model = AlexNet(x, keep_prob, num_classes=4, skip_layer=[])
score = model.fc8
softmax = tf.nn.softmax(score)

ckpt = tf.train.latest_checkpoint(checkpoints_path)
saver = tf.train.Saver()
print('load saved model:', ckpt)

fig = plt.figure(figsize=(30, 6))

with tf.Session() as sess:
    saver.restore(sess, ckpt)

    i = 0
    while input('input q to end:') != 'q' and i < len(image_list):
        img = cv2.resize(image_list[i].astype(np.float32), (227, 227))
        img = img.reshape((1, 227, 227, 3))
        pred_pro = sess.run(softmax, feed_dict={x: img, keep_prob: 1.})
        label = class_label_mapping[label_list[i]]
        class_name = class_label_mapping[np.argmax(pred_pro)]
        # cv2.imshow(path_list[i])
        image = Image.open(path_list[i])
        image.show()
        # fig.add_subplot(1, 10, i + 1)
        # plt.imshow(cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        print('------> label:{}   predict:{}    score:{}'.format(label, class_name, pred_pro))
        print('------> name:{}'.format(path_list[i].split('/')[-1]))
        i += 1
