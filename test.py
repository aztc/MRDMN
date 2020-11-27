

import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo import vision
from mxnet import init
from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler
from model import MRDM
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Test a model for fine-grained recognition.')
    parser.add_argument('--val-data-dir', type=str, default='data/birds_ori.rec',
                        help='validation pictures to use.')
    parser.add_argument('--num-class', type=str, default=200,
                        help='validation dataset classes.')
    parser.add_argument('--resolution', type=str, default=224,
                        help='validation resolution to use.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='test batch size per device (CPU/GPU).')
    parser.add_argument('--weights-path', type=str, default='weights',
                        help='weights path.')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')

    opt = parser.parse_args()
    return opt



def test():
    
    opt = parse_args()

    inputs = mx.sym.Variable('data')
    outputs = MRDM(inputs,num_class=opt.num_class,resolution=opt.resolution)
    net = gluon.SymbolBlock(outputs,inputs)
    
    contex = [mx.gpu(x) for x in range(opt.num_gpus)]
    net.load_parameters(opt.weights_path,ctx=contex)
    net.hybridize(static_alloc=True, static_shape=True)
    
    val_data_ori = mx.io.ImageRecordIter(
        path_imgrec         = opt.val_data_dir,
        preprocess_threads  = 6,
        shuffle             = False,
        resize              = 512,
        batch_size          = opt.batch_size,
        data_shape          = (3, 512, 512),
    )
    val_data_ori.reset()

    acc_top1 = mx.metric.Accuracy()
    
    for i,batch in enumerate(val_data_ori):
        data = gluon.utils.split_and_load(batch.data[0],ctx_list=contex,batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0],ctx_list=contex,batch_axis=0)
        logits = [net(X.astype('float32', copy=False)) for X in data]
        acc_top1.update(label,logits)
        
    name,acc =acc_top1.get()
    print('{}:{}'.format(name,acc))

if __name__ == '__main__':
    test()
