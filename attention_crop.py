from mxnet.symbol import Variable
from mxnet.symbol.contrib import BilinearResize2D
import mxnet as mx
from mxnet import nd
import numpy as np
from mxnet.symbol import SoftmaxOutput


class Att_Crop(mx.operator.CustomOp):
    def __init__(self,output_size):
        super(Att_Crop, self).__init__()
        self.outsize = output_size
        
        
    def forward(self, is_train, req, in_data, out_data, aux):
        fea = in_data[0]
        data = in_data[1]
        weights = in_data[2]
        prob = in_data[3]
        
        prob = prob / 3
        prob = nd.exp(prob)
        prob = prob/nd.sum(prob,axis=1,keepdims=1)
        
        w = nd.dot(prob,weights)
        w = nd.expand_dims(w,2)
        w = nd.expand_dims(w,3)     
        fea_w = fea*w
        
        d_w = data.shape[3]
        d_h = data.shape[2]
            
        w = fea.shape[2]
        n = fea.shape[0]
        
        
        fea = nd.mean(fea_w,axis=1,keepdims=1)
#        fea = nd.contrib.BilinearResize2D(fea,height=4*w,width=4*w)
#        w = 4*w
        
        max_val = nd.max(fea,axis=(2,3),keepdims=1)
        fea = fea / max_val
        
        
        if is_train:
            fea_mask = nd.greater_equal(fea,0.1)
            fea_mask2 = nd.greater_equal(fea,0.25)
        else:
            fea_mask = nd.greater_equal(fea,0.1)
            fea_mask2 = nd.greater_equal(fea,0.25)
        
        
        
        fea_mask1 = -nd.Pooling(-fea_mask,kernel=(5,5),pool_type='max',pad=(2,2))
        fea_mask1 = nd.Pooling(fea_mask1,kernel=(11,11),pool_type='max',pad=(5,5))
        cmask = nd.sum(fea_mask1,axis=(2,3),keepdims=1)
        cmask = nd.greater(fea,4)
        fea_mask = cmask * fea_mask2 * fea_mask1 + (1-cmask)*fea_mask2
        
        fea_mask = fea_mask[:,0,:,:].asnumpy()
        
        
        shape = self.outsize
        
        img_res = nd.zeros((n,3,shape,shape))
#        fea_res = nd.zeros((n,shape,shape))
        for i in range(n):
            m = fea_mask[i] 
            try:
                
                arg = np.float32(np.where(m==1))   
                ymin = np.int32(np.floor(np.min(arg[0])*(d_h/w)))
                ymax = np.int32(np.ceil(np.max(arg[0])*(d_h/w)))
                xmin = np.int32(np.floor(np.min(arg[1])*(d_w/w)))
                xmax = np.int32(np.ceil(np.max(arg[1])*(d_w/w)))
                
                x_center = (xmin+xmax)/2
                y_center = (ymin+ymax)/2
    #            
                x_length = xmax - xmin
                y_length = ymax - ymin
                longside = max(y_length,x_length)
                 
                x = np.int(max(x_center-longside/2,0))
                xmax = np.int(min(x_center+longside/2,d_w))
       
                l_x = xmax-x
                y = np.int(max(y_center-longside/2,0))
                ymax = np.int(min(y_center+longside/2,d_h))
                l_y = ymax-y
            
#            fea0 = fea[i]
#            fea0 = nd.expand_dims(fea0,0)
#            fea0 = nd.expand_dims(fea0,0)
#            fea0 = nd.contrib.BilinearResize2D(fea0,height=d_h,width=d_w)
#            
            
                img_crop = data[i,:,y:y+l_y,x:x+l_x]
            except:
                print(arg)
#            fea_crop = fea0[0,:,y:y+l_y,x:x+l_x]
            
            img_crop = nd.expand_dims(img_crop,0)
#            fea_crop  = nd.expand_dims(fea_crop,0)

            img_crop = nd.contrib.BilinearResize2D(img_crop,height=shape,width=shape)
#            fea_crop = nd.contrib.BilinearResize2D(fea_crop,height=shape,width=shape)
#                
#                if l_y > l_x:
#                    longside = int((l_y/l_x)*resize)
#                    img_crop = nd.contrib.BilinearResize2D(img_crop,height=longside,width=resize)
#                    s = int(np.floor((longside-shape)/2))
#                    img_crop = img_crop[:,:,s:s+shape,s1:s1+shape]
#                else:
#                    longside = int(l_x/l_y*resize)
#                    img_crop = nd.contrib.BilinearResize2D(img_crop,height=resize,width=longside) 
#                    s = int(np.floor((longside-shape)/2))
#                    img_crop = img_crop[:,:,s1:s1+shape,s:s+shape]
#                    
            
            img_res[i,:,:,:] = nd.squeeze(img_crop)
#            fea_res[i,:,:] = nd.squeeze(fea_crop)
#        fea_res = nd.expand_dims(fea_res,1)
#        img_res = img_res * fea_res
        self.assign(out_data[0], req[0], img_res)
#        self.assign(out_data[1], req[1], fea_res)
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = np.zeros(in_grad[0].shape)
        self.assign(in_grad[0], req[0], mx.nd.array(y))

@mx.operator.register("Att_Crop")
class Att_CropProp(mx.operator.CustomOpProp):
    def __init__(self,output_size):
        super(Att_CropProp, self).__init__(need_top_grad=False)
        self.outsize = output_size
    def list_arguments(self):
        return ['fea','data','weights','prob']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        fea_shape = in_shape[0]
        data_shape = in_shape[1]
        weights_shape = in_shape[2]
        prob_shape = in_shape[3]
        output_shape = (in_shape[0][0],3,int(self.outsize),int(self.outsize))
#        output_shape1 = (in_shape[0][0],int(self.outsize),int(self.outsize))
        return [fea_shape,data_shape,weights_shape,prob_shape], [output_shape], []
    def infer_type(self, in_type):
        datatype = in_type[0]
        return [datatype,datatype,datatype,datatype], [datatype], []
    def create_operator(self, ctx, shapes, dtypes):
        return Att_Crop(int(self.outsize))
    