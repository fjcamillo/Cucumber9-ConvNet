import tensorflow as tf
import numpy as np
import _pickle as cPickle
import time

with open('./inputs/data_batch_1') as f:
    batch1 = cPickle.load(f)
with open('./inputs/data_batch_2') as f:
    batch2 = cPickle.load(f)
with open('./inputs/data_batch_3') as f:
    batch3 = cPickle.load(f)
with open('./inputs/data_batch_4') as f:
    batch4 = cPickle.load(f)
with open('./inputs/data_batch_5') as f:
    batch5 = cPickle.load(f)
with open('./inputs/batches.meta') as f:
    meta = cPickle.load(f)

#Transforms the labels into their own dimensions
label_test = np.array(batch1['labels']).reshape(-1 ,1)
print(label_test.shape)

batch1['label_one_hot'] = [[1.0 if p==val else 0. for p in range(9)] for val in batch1['labels']]
batch2['label_one_hot'] = [[1.0 if p==val else 0. for p in range(9)] for val in batch2['labels']]
batch3['label_one_hot'] = [[1.0 if p==val else 0. for p in range(9)] for val in batch3['labels']]
batch4['label_one_hot'] = [[1.0 if p==val else 0. for p in range(9)] for val in batch4['labels']]
batch5['label_one_hot'] = [[1.0 if p==val else 0. for p in range(9)] for val in batch5['labels']]

dataset = {
    'features': [],
    'labels': []
}

for i in batch1['data']:
    dataset['features'].append(i)
for i in batch2['data']:
    dataset['features'].append(i)
for i in batch3['data']:
    dataset['features'].append(i)
for i in batch4['data']:
    dataset['features'].append(i)
for i in batch5['data']:
    dataset['features'].append(i)

for i in batch1['label_one_hot']:
    dataset['labels'].append(i)
for i in batch2['label_one_hot']:
    dataset['labels'].append(i)
for i in batch3['label_one_hot']:
    dataset['labels'].append(i)
for i in batch4['label_one_hot']:
    dataset['labels'].append(i)
for i in batch5['label_one_hot']:
    dataset['labels'].append(i)

dataset['features'] = np.array(dataset['features'])
dataset['labels'] = np.array(dataset['labels'])

training_feature = dataset['features'][:6000]
training_label = dataset['labels'][:2000]
test_feature = dataset['features'][6000:]
test_label = dataset['labels'][2000:]

training_conv = training_feature.reshape(2000,32,32,3)
testing_conv = test_feature.reshape(475,32,32,3)

def main(training_conv, testing_conv, training_label, test_label):
    with tf.device("/cpu:0"):
        x1 = tf.placeholder(tf.float32, [None, 32, 32, 3])
        yfinal = tf.placeholder(tf.float32, [None, 9])

        w_conv1 = tf.Variable(tf.truncated_normal([8, 8, 3, 12]), tf.float32)
        b_conv1 = tf.Variable(tf.ones([12]), tf.float32)

        w_conv2 = tf.Variable(tf.truncated_normal([4, 4, 12, 16]), tf.float32)
        b_conv2 = tf.Variable(tf.ones([16]))

        w1 = tf.Variable(tf.truncated_normal([32*32*16, 5000]), tf.float32)
        b1 = tf.Variable(tf.ones([5000]), tf.float32)

        w2 = tf.Variable(tf.truncated_normal([5000,9]), tf.float32)
        b2 = tf.Variable(tf.ones([9]), tf.float32)

        conv1 = tf.nn.relu(tf.nn.conv2d(x1, w_conv1, strides=[1, 1, 1, 1], padding="SAME")+b_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w_conv2, strides=[1, 1, 1, 1], padding="SAME")+b_conv2)

        converted = tf.reshape(conv2, [-1, 32*32*16])

        fc = tf.nn.relu(tf.matmul(converted, w1)+b1)
        fc2 = tf.matmul(fc, w2)+b2
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=yfinal, logits=fc2)
        train = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    epoch = 10000
    start = time.time()
    for i in range(epoch):
        if i%100==0:
            correct_prediction = tf.equal(tf.argmax(yfinal, 1), tf.argmax(fc2, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, {x1: testing_conv, yfinal:test_label}))
        sess.run(train, {x1: training_conv,yfinal:training_label})
    end = time.time() - start
    print("Total Time: {}".format(end))

    with open('./w1.txt', 'w') as weight_writer:
        weight_writer.write(sess.run(w1))




if __name__ == '__main__':
  main(training_conv, testing_conv, training_label, test_label)
