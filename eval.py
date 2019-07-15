import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg
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
x=tf.placeholder(tf.float32, shape=[1,224,224,3])
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(x, num_classes=num_classes, is_training=False, scope='resnet_v1_50')
    end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
    end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

with slim.arg_scope(vgg.vgg_arg_scope()):
    logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(x, num_classes=num_classes, is_training=False, scope='vgg_16')
end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])
model_checkpoint_map = {
'resnet_v1_50': os.path.join('./checkpoints/', 'resnet_v1_50','model.ckpt-49800'),
'vgg_16': os.path.join('./checkpoints/', 'vgg_16', 'vgg_16.ckpt')}
s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
r1=[]
r2=[]
d=[]
with tf.Session() as sess:
    s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
    s3.restore(sess, model_checkpoint_map['vgg_16'])
    for _,row in dev.iterrows():
        adv_image = Image.open(os.path.join(args.output_dir,row.filename), mode='r')
        raw_image = Image.open(os.path.join(args.input_dir,row.filename), mode='r')
        adv_image_arr = np.array(adv_image).astype(np.int32)
        raw_image_arr = np.array(raw_image).astype(np.int32)
        diff = adv_image_arr .reshape((-1, 3)) - raw_image_arr.reshape((-1, 3))
        distance = np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
        d.append(distance)
        adv_image = np.array(adv_image.resize([224,224],PIL.Image.BILINEAR)).astype(np.int32)
        adv_image = adv_image-np.array([123.68,116.78,103.94])
        adv_image = adv_image.reshape(1,224,224,3)
        a,b=sess.run([end_points_vgg_16['probs'],end_points_res_v1_50['probs']],feed_dict={x:adv_image})
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
print('vgg16',r1_score,len(dev)-len(result[result.targetedLabel==result.r1]))#,file=open("output.txt", "a+"))
print('resnet',r2_score,len(dev)-len(result[result.targetedLabel==result.r2]))
print('union',len(t))
print((r1_score+r2_score)/2)#,file=open("output.txt", "a+"))
