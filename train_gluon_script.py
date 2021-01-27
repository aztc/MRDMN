import time, logging, os, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo import vision
from mxnet import init
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler
from models.resnet import Resnet50_v2_dilat,Resnet18_v2_dilat,Resnet50_v1_dilat,Resnet18_v1_dilat
import os
from models.symbol_Resnet50_v2 import ResNet50_V2
from mxnet.image import RandomSizedCropAug
from mxnet.image import HorizontalFlipAug
import models.attention_net
from common.evaluate import Multi_Accuracy

class Options():
    def __init__(self,**kwargs):
       self.data_dir             = 'data'
       self.rec_train            = None
       self.rec_val              = None
       self.batch_size           = 128
       self.dtype                = 'float32'
       self.num_gpus             = 1
       self.num_epochs           = 100
       self.lr                   = 0.1
       self.momentum             = 0.9
       self.wd                   = 5e-4
       self.lr_mode              = 'step'
       self.lr_decay             = 0.1
       self.lr_decay_period      = 0
       self.lr_decay_epoch       = '30,60,90'
       self.warmup_lr            = 0
       self.warmup_epochs        = 0
       self.input_size           = 224
       self.jitter_param         = 0.4
       self.lighting_param       = 0.1
       self.max_random_area      = 1
       self.min_random_area      = 0.36
       self.max_aspect_ratio     = 0.1
       self.max_rotate_angle     = 20
       self.num_workers          = 20
       self.mean_rgb             = [0,0,0]
       self.num_classes          = 1000
       self.num_examples         = 1281167
       self.no_wd                = True
       self.save_frequency       = 50
       self.save_dir             = 'weights'
       self.resume_epoch         = 0
       self.resume_params        = None
       self.resume_states        = None
       self.log_interval         = 50
       self.logging_file         = None
       self.mode                 = 'hybrid'
       self.mixup                = False
       self.mixup_alpha          = 0.2
       self.mixup_off_epoch      = 0
       self.attention            = False
       self.att_size             = 224
       self.use_pretrain         = False
       self.union                = True
       self.distill              = True
       self.student              = None
       self.tea_net              = None
       self.att_net_params       = None
       self.model_name           = None
       self.KLLoss               = None
       
    def list_all_members(self):
        res = []
        for name,value in vars(self).items():
            res.append((name+' : '+str(value)))
        np.set_printoptions(threshold=2000)
        res = np.array(res)
        return res
       
opt = Options()



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
#os.environ["MXNET_CPU_WORKER_NTHREADS"]="20"

opt.data_dir        = r'data'
opt.rec_train       = 'birds-ori-train'
opt.rec_val         = 'birds-ori-test'
opt.batch_size      = 72
opt.num_gpus        = 2
opt.num_epochs      = 120
opt.num_classes     = 200
opt.num_examples    = 5994
opt.log_interval    = 25
opt.lr              = 0.05
opt.momentum        = 0
opt.wd              = 0
opt.lr_decay        = 0.1
opt.lr_decay_epoch  = '30,60,90'
opt.input_size      = 224
opt.logging_file    = 'logs_birds/birds_dilat14.log'
opt.save_frequency  = 50
opt.save_dir        = 'logs_birds/birds_dilat14'
opt.max_aspect_ratio = 0.1
opt.max_random_area  = 1
opt.min_random_area  = 0.36
opt.max_rotate_angle = 20
opt.jitter_param     = 0.4
opt.lighting_param   = 0.1
opt.mixup            = False
opt.mixup_alpha      = 0.3
opt.attention        = False
opt.att_size         = 224
opt.use_pretrain     = False
opt.union            = False
opt.distill          = False
opt.KLLoss           = False

if opt.attention:
    opt.model_name = model_name = 'Resnet50_v2'
else:
    opt.model_name = model_name = 'Resnet50_v1'
    
opt.student          = 'resnetv20_1'        
opt.tea_net          = r'init_weights/resnet50v2-birds-teanet-old-0000.params'
opt.att_net_params   = r'birds_dilat_old/0.8352-imagenet-ResNet50-best-0087.params'



filehandler = logging.FileHandler(opt.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(opt.list_all_members())

batch_size = opt.batch_size
classes = opt.num_classes
num_training_samples = opt.num_examples

num_gpus = opt.num_gpus

context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

lr_decay = opt.lr_decay
lr_decay_period = opt.lr_decay_period
if opt.lr_decay_period > 0:
    lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
else:
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
num_batches = num_training_samples // batch_size

lr_scheduler = LRSequential([
    LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
    LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                nepochs=opt.num_epochs - opt.warmup_epochs,
                iters_per_epoch=num_batches,
                step_epoch=lr_decay_epoch,
                step_factor=lr_decay, power=2)
])


kwargs = {'ctx': context, 'pretrained': True}


optimizer = 'SGD'
optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True


model_name = opt.model_name


if opt.use_pretrain:
    inputs = mx.sym.Variable('data')
    outputs = ResNet50_V2(inputs,classes=opt.num_classes)
    net = gluon.SymbolBlock(outputs,inputs)
    net.load_parameters(r'test_weights/0.8664-imagenet-Resnet50_v2-best-0072.params',ctx=context)

elif opt.union:
    from models.attention_net import Att_Master
    inputs0 = mx.sym.Variable('data')
    inputs1 = mx.sym.Variable('data1',shape=(-1,200))
    outputs = Att_Master([inputs0,inputs1],num_class=opt.num_classes)
    net = gluon.SymbolBlock(outputs,[inputs0,inputs1])
    net.initialize(init.Xavier(), ctx = context)
    net.load_parameters(r'union_weights/resnet50v2-teanet-0000.params',ctx=context,allow_missing=False,ignore_extra=False)

elif opt.distill:
    from models.symbol_Resnet50_v2 import ResNet50_V2
    inputs = mx.sym.Variable('data')
    outputs = ResNet50_V2(inputs,classes=classes,model_name=opt.student)
    net = gluon.SymbolBlock(outputs,inputs)
    net.load_parameters(opt.tea_net,ctx=context,allow_missing=False,ignore_extra=True)
    net.collect_params().reset_ctx(context)
    
elif opt.attention:
    net = get_model(model_name, **kwargs)
    with net.name_scope():
        net.output = nn.Dense(classes)
    net.output.initialize(init.Xavier(), ctx = context)
    net.collect_params().reset_ctx(context)
    
else:
#    from models.symbol_Resnet50_v1 import ResNet50_V1_dilat
#    inputs = mx.sym.Variable('data')
#    outputs = ResNet50_V1_dilat(inputs,classes=classes)
#    net = gluon.SymbolBlock(outputs,inputs)
#####    net.initialize(init.Xavier(), ctx = context)
#    net.initialize(ctx=context)
##    net.load_parameters(r'init_weights/resnet50_v1_U-0000.params',ctx=context,allow_missing=True,ignore_extra=True)
#    net.load_parameters(r'init_weights/0.7722-Resnet50_v1_U-best-0014.params',ctx=context,allow_missing=True,ignore_extra=True)
#
###    
#    net = get_model('Resnet50_v1', pretrained=True)
    net = Resnet50_v1_dilat(pretrained=True,ctx=context)
    with net.name_scope():
        net.output = nn.Dense(classes)
    net.output.initialize(init.Xavier(), ctx = context)
    net.collect_params().reset_ctx(context)
    
    

net.cast(opt.dtype)
if opt.resume_params is not None:
    net.load_parameters(opt.resume_params, ctx = context)

if opt.attention:
    from models.symbol_Resnet50_v1 import ResNet50_V1_dilat
    inputs = mx.sym.Variable('data')
    x = mx.sym.contrib.BilinearResize2D(inputs,height=224,width=224)
    x,att_fea,weight = ResNet50_V1_dilat(x,classes=opt.num_classes)
    data_crop = mx.sym.Custom(att_fea,inputs,weight,x,output_size=336,op_type='Att_Crop')
    outputs = mx.sym.Group([x,data_crop])
    att_net = gluon.SymbolBlock(outputs,inputs)
    att_net.load_parameters(opt.att_net_params,ctx=context)
    att_net.hybridize(static_alloc=True, static_shape=True)
else:
    att_net=None
    
    
if opt.distill:
    from models.attention_net import Att_Multi_Scale
    inputs = mx.sym.Variable('data')
    outputs = Att_Multi_Scale(inputs,num_class=opt.num_classes)
    tea_net = gluon.SymbolBlock(outputs,inputs)
    tea_net.load_parameters(opt.tea_net,ctx=context)
    tea_net.hybridize(static_alloc=True, static_shape=True)
    
    
    
# Two functions for reading data from record file or raw images
def get_data_rec(opt):
    
    rec_train = os.path.join(opt.data_dir,opt.rec_train+'.rec')
    rec_train_idx = os.path.join(opt.data_dir,opt.rec_train+'.idx')
    rec_val = os.path.join(opt.data_dir,opt.rec_val+'.rec')
    rec_val_idx = os.path.join(opt.data_dir,opt.rec_val+'.idx')

    input_size = opt.input_size
    crop_ratio = 1
    resize = int(math.ceil(input_size / crop_ratio))

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label
    

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = opt.num_workers,
        shuffle             = True,
        batch_size          = opt.batch_size,

        data_shape          = (3, input_size, input_size),
        mean_r              = opt.mean_rgb[0],
        mean_g              = opt.mean_rgb[1],
        mean_b              = opt.mean_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = opt.max_aspect_ratio,
        max_random_area     = opt.max_random_area,
        min_random_area     = opt.min_random_area,
        max_rotate_angle    = opt.max_rotate_angle,
        brightness          = opt.jitter_param,
        saturation          = opt.jitter_param,
        contrast            = opt.jitter_param,
        pca_noise           = opt.lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = opt.num_workers,
        shuffle             = False,
        rand_crop           = False,
        batch_size          = opt.batch_size,
        resize              = resize,
        data_shape          = (3, input_size, input_size),
        mean_r              = opt.mean_rgb[0],
        mean_g              = opt.mean_rgb[1],
        mean_b              = opt.mean_rgb[2],
    )
    return train_data, val_data, batch_fn



train_data, val_data, batch_fn = get_data_rec(opt)


if opt.mixup:
    train_metric = mx.metric.RMSE()
else:   
    train_metric = mx.metric.Accuracy()
    
    
if opt.attention and opt.union:
  acc_top1 = Multi_Accuracy(num=2,mode=3)
elif opt.attention:
  acc_top1 = Multi_Accuracy(num=2,mode=0)
else:
  acc_top1 = mx.metric.Accuracy()  


save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0



def mixup_transform(label, classes, lam=1, eta=0.0):
    if isinstance(label, nd.NDArray):
        label = [label]
    res = []
    for l in label:
        y1 = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
        y2 = l[::-1].one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
        res.append(lam*y1 + (1-lam)*y2)
    return res
    
    
def test(ctx, val_data, opt, att_net=None):
    
    val_data.reset()
    acc_top1.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        
        if att_net is not None:
            att_outputs = [att_net(X.astype(opt.dtype, copy=False)) for X in data]
            att_logist = [X[0] for X in att_outputs]
            data = [X[1] for X in att_outputs]
        
        if opt.union:
            outputs =[]
            for idx in range(len(data)):
                outputs.append(net(data[idx],att_logist[idx]))
            
            preds_concat=[]
            for i in range(len(outputs)):
                preds=[]
                preds.append(outputs[i][0])
                preds.append(outputs[i][1])
                preds_concat.append(preds)   
            acc_top1.update(label, preds_concat)
        
        elif att_net is not None:
            data = [nd.contrib.BilinearResize2D(X,height=opt.att_size,width=opt.att_size) for X in data]
            outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            preds_concat=[]
            for i in range(len(outputs)):
                preds=[]
                preds.append(outputs[i])
                preds.append(att_logist[i])
                preds_concat.append(preds)   
            acc_top1.update(label, preds_concat)
        else:
            outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            acc_top1.update(label, outputs)
            
            
    name, top1 = acc_top1.get()
    
    if type(name)==str:
        name = [name]
        top1 = [top1]
    return (name, top1)



ctx = context
if opt.mode == 'hybrid':
    net.hybridize(static_alloc=True, static_shape=True)

if isinstance(ctx, mx.Context):
    ctx = [ctx]
if opt.resume_params is '':
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

if opt.no_wd:
    for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
if opt.resume_states is not None:
    trainer.load_states(opt.resume_states)



if opt.mixup:
    sparse_label_loss = False
else:
    sparse_label_loss = True
      


if opt.distill:
    L_dis = gcv.loss.DistillationSoftmaxCrossEntropyLoss(temperature=3,
                                                     hard_weight=0.5,
                                                     sparse_label=sparse_label_loss) 
      
L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)
KLDivLoss = gluon.loss.KLDivLoss(weight=1)

best_val_score = 0

for epoch in range(opt.resume_epoch, opt.num_epochs):
#    if epoch == 0:
#        val_data.reset()    
#        top1_name, top1_val = test(ctx, val_data, opt, att_net)  
#        for idx in range(len(top1_name)):
#            name = top1_name[idx]
#            val  = top1_val[idx]
#            logger.info('Epoch[%d] Validation-%s=%.4f', epoch, name, val)
#    
    
    tic = time.time()

    train_data.reset()
    train_metric.reset()
    btic = time.time()
       
    for i, batch in enumerate(train_data):
        data, label = batch_fn(batch, ctx)
        
        
        if opt.attention:
            att_outputs = [att_net(X.astype(opt.dtype, copy=False)) for X in data]
            att_logist = [X[0] for X in att_outputs]
            data = [X[1] for X in att_outputs]
        
            
        if opt.mixup:
            lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
            if epoch >= opt.num_epochs - opt.mixup_off_epoch:
                lam = 1
            data = [lam*X + (1-lam)*X[::-1] for X in data]

            eta = 0.0
            label = mixup_transform(label, classes, lam, eta)
        
        
        if opt.distill:
            tea_outputs = [tea_net(X.astype(opt.dtype, copy=False)) for X in data]
            tea_prob = [nd.softmax(X/3) for X in tea_outputs]
            
        with ag.record():
            
            if opt.union:
                outputs=[]
                for idx in range(len(data)):
                    outputs.append(net(data[idx],att_logist[idx]))
            
            elif att_net is not None:
                data = [nd.contrib.BilinearResize2D(X,height=opt.att_size,width=opt.att_size) for X in data]
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            
            else:
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            
            if opt.distill:  
                loss = [L_dis(yhat, y.astype(opt.dtype, copy=False), p) for yhat, y, p in zip(outputs, label, tea_prob)]
            elif opt.union:
                l1 = [L(yhat[0], y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                l2 = [L(yhat[1], y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                loss = [x+y for x,y in zip(l1,l2)]
            elif opt.KLLoss:
                l1 = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                outputs1 = [X/3 for X in outputs]
                att_logist1 = [X/3 for X in att_logist]
                l2 = [KLDivLoss(yhat, y) for yhat, y in zip(outputs1,att_logist1)]
                loss = [x-y for x,y in zip(l1,l2)]
            else:
                loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()
        trainer.step(batch_size)

        if opt.mixup and opt.union:
            output_softmax = [nd.SoftmaxActivation(out[-1].astype('float32', copy=False)) \
                            for out in outputs]
            train_metric.update(label, output_softmax)
        
        elif opt.mixup:
            output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                            for out in outputs]
            train_metric.update(label, output_softmax)
        
        else:
            train_metric.update(label, outputs)

        if opt.log_interval and not (i+1)%opt.log_interval:
            train_metric_name, train_metric_score = train_metric.get()
            logger.info('Epoch[%d] Batch [%d]  Speed=%.1f Hz  %s=%.3f '%(
                        epoch, i, batch_size*opt.log_interval/(time.time()-btic),
                        train_metric_name, train_metric_score))
            btic = time.time()

    train_metric_name, train_metric_score = train_metric.get()
    throughput = int(batch_size * i /(time.time() - tic))

    val_data.reset()    

    top1_name, top1_val = test(ctx, val_data, opt, att_net)  
    
    logger.info('Epoch[%d] Train-accuracy=%f'%(epoch, train_metric_score))
    logger.info('Epoch[%d] speed: %d samples/sec\ttime cost: %f'%(epoch, throughput, time.time()-tic))
    

    for idx in range(len(top1_name)):
        name = top1_name[idx]
        val  = top1_val[idx]
        logger.info('Epoch[%d] Validation-%s=%.4f', epoch, name, val)
    
    if len(top1_val) > 0:
        top1_val = top1_val[-1]
        
    if top1_val > best_val_score:
        best_val_score = top1_val
        net.export('%s/%.4f-%s-best'%(save_dir, best_val_score, model_name),epoch)
#            trainer.save_states('%s/%.4f-imagenet-%s-%d-best.states'%(save_dir, best_val_score, model_name, epoch))

    if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
        net.export('%s/%s'%(save_dir, model_name),epoch)
#            trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, model_name, epoch))

if save_frequency and save_dir:
    net.export('%s/%s'%(save_dir, model_name),opt.num_epochs-1)
#        trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, model_name, opt.num_epochs-1))




