import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1,inception
import inception_resnet_v2
import vgg
import numpy as np
#from scipy.misc import imread
import pandas as pd
from PIL import Image
import sys
import argparse

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

slim = tf.contrib.slim
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--alpha', help='per step decrease',type=float, default=300)
parser.add_argument('-i', '--num_iter', help='num_iter', type=int, default=40)
parser.add_argument('-rw', '--resnet_weight',type=float, default=0.8)
parser.add_argument('-vw', '--vgg_weight',type=float, default=1.0)
parser.add_argument('-c', '--confidence',type=float, default=0.5)
parser.add_argument('-d','--weight_decay',type=float, default=1.0)
parser.add_argument('-mm','--min_max_range',type=float, default=0.3)
parser.add_argument("-ir", "--is_round", help="round x per round",action="store_true")
parser.add_argument('-mina','--mina',type=float, default=100.0)
parser.add_argument('-maxa','--maxa',type=float, default=400.0)
parser.add_argument("-iw", "--is_wrong", help="wrong",action="store_true")
parser.add_argument("-norm", "--norm", help="use norm",action="store_true")
parser.add_argument("-im", "--is_mask", help="use mask",action="store_true")
parser.add_argument('-ms','--mask_size',type=int, default=5)

parser.add_argument('--input_dir',type=str, default='./dev_data/')
parser.add_argument('--output_dir',type=str, default='./output/')
parser.add_argument('--dev_dir',type=str, default='./dev_data/')

args = parser.parse_args()
print(args,file=open("output.txt", "a+"))
batch_shape=[1,299,299,3]
num_iter= args.num_iter
model_checkpoint_map = {
    'resnet_v1_50': os.path.join('./checkpoints/', 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join('./checkpoints/', 'vgg_16', 'vgg_16.ckpt'),
    'InceptionV3': os.path.join('./checkpoints/', 'inception_v3', 'inception_v3.ckpt'),
}

dev=pd.read_csv(os.path.join(args.dev_dir,'dev.csv'))

if args.is_wrong:
    try:
        wrong_list=pd.read_csv('./wrong_list.csv',header=None)
        dev=dev[dev.filename.isin(wrong_list[0])].reset_index(drop=True)
    except:
        sys.exit(0)

def Graph(x, y,raw_image):
    num_classes=110
    batch_size=1
    weight = [args.resnet_weight,args.vgg_weight]
    x_int=x/255*2-1
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits_inception_v3, end_points_inception_v3 = inception.inception_v3(x_int, num_classes=110,reuse=tf.AUTO_REUSE, is_training=False, scope='InceptionV3')
    one_hot = tf.one_hot(y, num_classes)
    logits = logits_inception_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,logits,label_smoothing=0.0,weights=1.0)
    grad = tf.gradients([cross_entropy], [x])[0]
    if args.norm:
        grad = grad/tf.norm(grad)
    else:
        #grad = grad/tf.reshape(tf.norm(grad,axis=-1),[1,299,299,1])
        grad = tf.transpose(grad,[0,3,1,2])
        grad = grad/tf.reshape(tf.norm(tf.reshape(grad,[batch_size,3,-1]),axis=2),[batch_size,3,1,1])
        grad = tf.transpose(grad,[0,2,3,1])
        
    if args.is_mask:
        mask = tf.ones(shape=[int(299-2*args.mask_size),int(299-2*args.mask_size),3])
        mask = tf.pad(mask,tf.constant([[args.mask_size,args.mask_size],[args.mask_size,args.mask_size],[0,0]]))
        grad = grad*mask

    alpha = args.maxa
    x = x - alpha * grad#*tf.concat([tf.ones([299,299,1]),tf.ones([299,299,1]),tf.zeros([299,299,1])],-1)
    x = tf.clip_by_value(x, 0, 255)
    out_x = x-raw_image
    out_x = tf.floor(tf.abs(out_x))*tf.sign(out_x)+raw_image
    out_x =  tf.round(tf.clip_by_value(out_x, 0, 255))
    return x,out_x

def Eval(x_img,y):
    input_image=2*x_img/255-1
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits_inception_v3, end_points_inception_v3 = inception.inception_v3(input_image, num_classes=110, is_training=False, scope='InceptionV3',reuse=tf.AUTO_REUSE)
        inc_label=tf.argmax(end_points_inception_v3['Predictions'][0],-1)
        y_inc=end_points_inception_v3['Predictions'][0][y[0]]
    return inc_label,y_inc

# Define Graph
x=tf.placeholder(tf.float32, shape=[1,299,299,3])
y=tf.placeholder(tf.int32, shape=1)
raw_image_placeholder=tf.placeholder(tf.float32, shape=[1,299,299,3])
x_adv,out_x = Graph(x,y,raw_image_placeholder)
s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
x_img=tf.placeholder(tf.float32, shape=[1,299,299,3])
inc_label,y_inc=Eval(x_img,y)


with tf.Session(config=config) as sess:
    s1.restore(sess, model_checkpoint_map['InceptionV3'])
    success=0
    d_counter=0
    for _,row in dev.iterrows():
        d_counter+=1
        raw_image = np.array(Image.open(os.path.join(args.input_dir,row.filename), mode='r'))
        raw_image = raw_image.reshape(1,299,299,3)
        output = raw_image
        output_image = Image.fromarray(output.astype(np.uint8)[0])
        eval_input=np.array(output_image).astype(np.float32).reshape([1,299,299,3])
        inc_l,inc_pred=sess.run([inc_label,y_inc],feed_dict={x_img:eval_input,y:[row.targetedLabel]})
        output = raw_image
        for _ in range(num_iter):
            output = np.array(output).astype(np.float32).reshape(1,299,299,3)
            output,output_image = sess.run([x_adv,out_x],feed_dict={x:output,y:[row.targetedLabel],
                                                        raw_image_placeholder:raw_image})
            #output_image = Image.fromarray(np.round(output_image).astype(np.uint8)[0])
            output_image = Image.fromarray(output_image.astype(np.uint8)[0])
            eval_input=np.array(output_image).astype(np.float32).reshape([1,299,299,3])
            inc_l,inc_pred=sess.run([inc_label,y_inc],feed_dict={x_img:eval_input,y:[row.targetedLabel]})
            if (inc_l==row.targetedLabel):
                success+=1
                output_image.save(os.path.join(args.output_dir,row.filename), format='PNG')
                message = '{}/{} {} {}'.format(success,d_counter,inc_l,row.targetedLabel).ljust(50)
                print(message,end='\r')
                break
            else:
                message = '{}/{} {} {}'.format(success,d_counter,inc_l,row.targetedLabel).ljust(50)
                print(message,end='\r')
        else:
            output_image.save(os.path.join(args.output_dir,row.filename), format='PNG')
print(end='\n')

                
            
