import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import vgg
import numpy as np
import pandas as pd
import sys
import argparse
from PIL import Image
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

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

def Graph(x, y, res_weight,vgg_weight,inc_weight,y_pred_res,y_pred_vgg,y_pred_inc,raw_image):
    num_classes=110
    batch_size=1
    weight = [args.resnet_weight,args.vgg_weight]
    bias = tf.reshape(tf.constant([123.68,116.78,103.94]),[1,1,1,3])
    x_int = tf.image.resize_bilinear(x, [224,224],align_corners=True)

    x_int = x_int-bias

    with slim.arg_scope(resnet_v1.resnet_arg_scope()) as scope:
        logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(x_int, num_classes=num_classes, is_training=False, scope='resnet_v1_50',reuse=tf.AUTO_REUSE)
        end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])

    with slim.arg_scope(vgg.vgg_arg_scope()) as scope:
        logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(x_int, num_classes=num_classes, is_training=False, scope='vgg_16',reuse=tf.AUTO_REUSE)
        end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
        
    one_hot = tf.one_hot(y, num_classes)

    sum_prob= tf.clip_by_value((res_weight*y_pred_res+ vgg_weight*y_pred_vgg),0,args.confidence)
    logits_resnet = end_points_res_v1_50['logits']
    logits_vgg = end_points_vgg_16['logits']
    
    logits = res_weight*logits_resnet+ vgg_weight*logits_vgg
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,logits,label_smoothing=0.0,weights=1.0)
    grad = tf.gradients([cross_entropy], [x])[0]
    #grad = tf.layers.dropout(grad,noise_shape=[1,299,299,3])


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
    
    alpha = args.maxa-(args.maxa-args.mina)/(args.confidence)*sum_prob
    x = x - alpha * grad#*tf.concat([tf.ones([299,299,1]),tf.ones([299,299,1]),tf.zeros([299,299,1])],-1)
    x = tf.clip_by_value(x, 0, 255)
    out_x = x-raw_image
    out_x = tf.floor(tf.abs(out_x))*tf.sign(out_x)+raw_image
    out_x =  tf.round(tf.clip_by_value(out_x, 0, 255))
    
    return x,out_x

def Eval(x_img_224,x_img_299,y):

    input_image=x_img_224-tf.reshape(tf.constant([123.68,116.78,103.94]),[1,1,1,3])

    with slim.arg_scope(resnet_v1.resnet_arg_scope()) as scope:
        logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(input_image, num_classes=110, is_training=False, scope='resnet_v1_50',reuse=tf.AUTO_REUSE)
        end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
        end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])
        res_label=tf.argmax(end_points_res_v1_50['probs'][0],-1)
        y_r=end_points_res_v1_50['probs'][0][y[0]]

    with slim.arg_scope(vgg.vgg_arg_scope()) as scope:
        logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(input_image, num_classes=110, is_training=False, scope='vgg_16',reuse=tf.AUTO_REUSE)
        end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
        end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])
        vgg_label=tf.argmax(end_points_vgg_16['probs'][0],-1)
        y_v=end_points_vgg_16['probs'][0][y[0]]

    return res_label,vgg_label,y_r,y_v

# Define Graph
x=tf.placeholder(tf.float32, shape=[1,299,299,3])
y=tf.placeholder(tf.int32, shape=1)
res_weight = tf.placeholder(tf.float32, shape=1)
vgg_weight = tf.placeholder(tf.float32, shape=1)
y_pred_res = tf.placeholder(tf.float32, shape=None)
y_pred_vgg = tf.placeholder(tf.float32, shape=None)
raw_image_placeholder=tf.placeholder(tf.float32, shape=[1,299,299,3])
x_adv,out_x = Graph(x,y,res_weight,vgg_weight,y_pred_res,y_pred_vgg,raw_image_placeholder)
s1 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
s2 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
x_img=tf.placeholder(tf.float32, shape=[1,224,224,3])
res_label,vgg_label,y_res,y_vgg=Eval(x_img,y)


with tf.Session(config=config) as sess:
    s1.restore(sess, model_checkpoint_map['resnet_v1_50'])
    s2.restore(sess, model_checkpoint_map['vgg_16'])
    success=0
    d_counter=0
    for _,row in dev.iterrows():
        d_counter+=1
        raw_image = np.array(Image.open(os.path.join(args.input_dir,row.filename), mode='r'))
        raw_image = raw_image.reshape(1,299,299,3)
        output = raw_image
        output_image = Image.fromarray(output.astype(np.uint8)[0])
        eval_input=np.array(output_image.resize([224,224],Image.BILINEAR)).astype(np.float32).reshape([1,224,224,3])
        res_l,vgg_l,res_pred,vgg_pred=sess.run([res_label,vgg_label,y_res,y_vgg],feed_dict={x_img:eval_input,y:[row.targetedLabel]})
        #print(type(res_pred),res_pred)
        output = raw_image
        iter_res_weight=0.5
        iter_vgg_weight=0.5
        for _ in range(num_iter):
            output = np.array(output).astype(np.float32).reshape(1,299,299,3)
            output,output_image = sess.run([x_adv,out_x],feed_dict={x:output,y:[row.targetedLabel],
                                                        res_weight:[iter_res_weight],
                                                        vgg_weight:[iter_vgg_weight],
                                                        y_pred_res:[res_pred],
                                                        y_pred_vgg:[vgg_pred],
                                                        raw_image_placeholder:raw_image})
            #output_image = Image.fromarray(np.round(output_image).astype(np.uint8)[0])
            output_image = Image.fromarray(output_image.astype(np.uint8)[0])
            eval_input=np.array(output_image.resize([224,224],Image.BILINEAR)).astype(np.float32).reshape([1,224,224,3])
            res_l,vgg_l,res_pred,vgg_pred=sess.run([res_label,vgg_label,y_res,y_vgg],feed_dict={x_img:eval_input,y:[row.targetedLabel]})
            if (res_l==row.targetedLabel)&(vgg_l==row.targetedLabel):
                success+=1
                output_image.save(os.path.join(args.output_dir,row.filename), format='PNG')
                message = '{}/{} {} {} {}'.format(success,d_counter,res_l,vgg_l,row.targetedLabel).ljust(50)
                print(message,end='\r')
                break
            else:
                message = '{}/{} {} {} {}'.format(success,d_counter,res_l,vgg_l,row.targetedLabel).ljust(50)
                print(message,end='\r')

            if (res_l==row.targetedLabel)&(vgg_l!=row.targetedLabel):
                iter_res_weight=0
                iter_vgg_weight=1.0
            elif (res_l!=row.targetedLabel)&(vgg_l==row.targetedLabel):
                iter_res_weight=1.0
                iter_vgg_weight=0
            else:
                iter_res_weight=0.5
                iter_vgg_weight=0.5
        else:
            #output_image = Image.fromarray(output.astype(np.uint8)[0])
            output_image.save(os.path.join(args.output_dir,row.filename), format='PNG')
print(end='\n')

                
            
