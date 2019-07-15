import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import inception_resnet_v2
from tensorflow.contrib.slim.nets import inception
import numpy as np
import pandas as pd
from PIL import Image
import PIL

import argparse
import sys

slim = tf.contrib.slim
parser = argparse.ArgumentParser()

parser.add_argument('--input_dir',type=str, default='./dev_data/')
parser.add_argument('--output_dir',type=str, default='./output/')
parser.add_argument('--dev_dir',type=str, default='./dev_data/')
parser.add_argument("-iw", "--is_wrong", help="wrong",action="store_true")
args = parser.parse_args()
dev=pd.read_csv(os.path.join(args.dev_dir,'dev.csv'))
if args.is_wrong:
    try:
        wrong_list=pd.read_csv('./wrong_list.csv',header=None)
        dev=dev[dev.filename.isin(wrong_list[0])].reset_index(drop=True)
    except:
        sys.exit(0)
else:
    try:
        wrong_list=pd.read_csv('./wrong_list.csv',header=None)
        t_dev=dev[dev.filename.isin(wrong_list[0])].reset_index(drop=True)
        if len(t_dev)==0: sys.exit(0) 
    except:
        pass

num_classes=110
batch_size=1
x=tf.placeholder(tf.float32, shape=[1,299,299,3])
x_input=x/255*2-1

with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits_inception_v3, end_points_inception_v3 = inception.inception_v3(x_input, num_classes=110, is_training=False, scope='InceptionV3')

with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_inception_resnet_v2, end_points_inception_resnet_v2 = inception_resnet_v2.inception_resnet_v2(x_input, num_classes=110, is_training=False, scope='InceptionResnetV2')


model_checkpoint_map = {
    'resnet_v1_50': os.path.join('./checkpoints/', 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join('./checkpoints/', 'vgg_16', 'vgg_16.ckpt'),
    'InceptionV3': os.path.join('./checkpoints/', 'inception_v3', 'inception_v3.ckpt'),
    'InceptionResnetV2': os.path.join('./checkpoints/', 'inception_resnet_v2', 'model.ckpt-193019'),
}
s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))

r1=[]
r2=[]
d=[]

with tf.Session() as sess:
    s2.restore(sess, model_checkpoint_map['InceptionV3'])
    s3.restore(sess, model_checkpoint_map['InceptionResnetV2'])
    for _,row in dev.iterrows():
        adv_image = Image.open(os.path.join(args.output_dir,row.filename), mode='r')
        raw_image = Image.open(os.path.join(args.input_dir,row.filename), mode='r')
        adv_image_arr = np.array(adv_image).astype(np.int32)
        raw_image_arr = np.array(raw_image).astype(np.int32)
        diff = adv_image_arr .reshape((-1, 3)) - raw_image_arr.reshape((-1, 3))
        distance = np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
        d.append(distance)
        adv_image = np.array(adv_image).astype(np.float32).reshape(1,299,299,3)
        a,b=sess.run([logits_inception_v3,logits_inception_resnet_v2],feed_dict={x:adv_image})
        r1.append(a)
        r2.append(b)
r1=np.concatenate(r1)
r2=np.concatenate(r2)
r1=r1.argmax(1)
r2=r2.argmax(1)
result=pd.DataFrame({'r1':r1,'r2':r2,'d':d})
result=dev.join(result)
r1_score=result[result.targetedLabel==result.r1].d.mean()*len(result[result.targetedLabel==result.r1])/len(dev)+64*(1-len(result[result.targetedLabel==result.r1])/len(dev))
r2_score=result[result.targetedLabel==result.r2].d.mean()*len(result[result.targetedLabel==result.r2])/len(dev)+64*(1-len(result[result.targetedLabel==result.r2])/len(dev))
result[~((result.targetedLabel==result.r1)&(result.targetedLabel==result.r2))].filename.to_csv('wrong_list.csv',index=False)
#wrong_list=pd.read_csv('./wrong_list.csv',header=None)[0]
t=result[(result.targetedLabel==result.r1)&(result.targetedLabel==result.r2)]
print(t.d.mean())#,file=open("output.txt", "a+"))
print('logits_inception_v3',r1_score,len(dev)-len(result[result.targetedLabel==result.r1]))#,file=open("output.txt", "a+"))
print('logits_inception_resnet_v2',r2_score,len(dev)-len(result[result.targetedLabel==result.r2]))
print('union',len(t))
print((r1_score+r2_score)/2)#,file=open("output.txt", "a+"))
