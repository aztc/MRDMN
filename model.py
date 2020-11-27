from attention_net import atten_net
from detail_net import detai_net
import mxnet as mx
from attention_crop import Att_Crop

inputs = mx.sym.Variable('data')

def MRDM(inputs,num_class=200,resolution=224):
    inputs224 = mx.sym.contrib.BilinearResize2D(inputs,height=224,width=224)
    att_logits, att_fea, weight = atten_net(inputs224,num_class)
    data_crop = mx.sym.Custom(att_fea,inputs,weight,att_logits,output_size=336,op_type='Att_Crop')
    data_crop = mx.sym.contrib.BilinearResize2D(data_crop,height=resolution,width=resolution)
    detail_logits = detai_net(data_crop,num_class,model_name='resnetv20_224')
    logits = 0.25*att_logits + 0.75*detail_logits    
    return logits
    
    
    
