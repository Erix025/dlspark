from dlspark.auto_grad import Tensor, TensorOp, NDArray
import numpy

# TODO: Implement Conv2D, MaxPool2D, AvgPool2D

def img2col(x, kernel_size, stride, keep_channel=False):
    """把一张图中C个需要卷积的区域拉成一个列向量
    Args:
        x (N,C,H,W): 
        kernel_size (O,C,H,W): 
        stride (int):
        keep_channel (bool): 如果为True则保留channel维度，否则把所有channel的数据合并
    Returns:
        x_col: (N*H_out*W_out, C*kernel_size*kernel_size)
    """
    # 
    N, C, H, W = x.shape
    x_col = []
    for i in range(0, H - kernel_size + 1, stride):
        for j in range(0, W - kernel_size + 1, stride):
            if keep_channel:
                col = x[:, :, i:i + kernel_size, j:j + kernel_size].reshape(N*C, -1)
            else:
                col = x[:, :, i:i + kernel_size, j:j + kernel_size].reshape(N, -1)
            x_col.append(col)
    if keep_channel:
        # H_out*W_out, N*C, kernel*kernel -> N*C*H_out*W_out, kernel*kernel
        x_col = numpy.array(x_col).transpose(1,0,2).reshape(-1, kernel_size*kernel_size)
    else:
        # H_out*W_out, N, C*kernel*kernel -> N*H_out*W_out, C*kernel*kernel
        x_col = numpy.array(x_col).transpose(1,0,2).reshape(-1, C*kernel_size*kernel_size)
    return x_col

class Conv2D(TensorOp):
    def __init__(self, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
    
    def compute(self, x: NDArray, kernel: NDArray, bias:NDArray) -> NDArray:
        N, in_channels, in_height, in_width = x.shape
        out_channels, _, kernel_height, kernel_width = kernel.shape
        
        assert kernel_height == kernel_width, "Kernel height and width must be the same"
        
        H_out = (in_height - kernel_height + 2 * self.padding) // self.stride + 1
        W_out = (in_width - kernel_width + 2 * self.padding) // self.stride + 1
        
        out = numpy.zeros((N, out_channels, H_out, W_out), dtype=x.dtype)
        x_padded = numpy.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        out = self.__conv__(x_padded, kernel, bias, self.stride)
        
        return out
    
    def __conv__(self, x, kernel, bias=None, stride=1):
        # x 是已经经过padding处理好的矩阵 
        N, C, H, W = x.shape
        O, _, kernel_size, _ = kernel.shape
        cols = img2col(x, kernel_size, stride)
        output = numpy.dot(cols, kernel.reshape(O, -1).T)
            
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        
        if bias is not None and bias.any():
            output += bias
        return output.reshape(N, H_out, W_out, O).transpose(0, 3, 1, 2)
    
    def gradient(self, out_grad, node):
        x, kernel, bias = node.inputs
        x = x.numpy()
        kernel = kernel.numpy()
        bias = bias.numpy()
        out_grad = out_grad.numpy()
        N, C, H, W = x.shape
        O, _ , kernel_size, _ = kernel.shape
        H_out, W_out = out_grad.shape[2], out_grad.shape[3]
        s = self.stride
        
        dw = numpy.zeros_like(kernel, dtype=x.dtype)
        dy_pad_shape = (N, O, 
            2 * kernel_size + (H_out - 1) * s - 1,
            2 * kernel_size + (W_out - 1) * s - 1) 
        # 计算传给前一层的grad时用到，作为被卷积的矩阵
        dy_shape = (N, O, (H_out - 1) * s + 1, (W_out - 1) * s + 1)
        # 计算dw时用到，间隔stride-1填充0。作为卷积核
        grad_padded = numpy.zeros(dy_pad_shape,dtype=x.dtype)
        grad = numpy.zeros(dy_shape, dtype=x.dtype)
        for i in range(H_out):
            for j in range(W_out):
                grad[:,:,i * s, j * s] = out_grad[:,:,i,j]
                grad_padded[:,:,kernel_size + i * s - 1, kernel_size + j * s - 1] = out_grad[: , : ,i, j]
        # 处理 padding 和 stride
        da = self.__conv__(grad_padded, kernel=kernel[:,:,::-1, ::-1].transpose(1,0,2,3))
        # N,out,H_pad,W_pad 和 in,out,K,K 卷积完是 N,in,H,W
        dw = self.__conv__(x.transpose(1,0,2,3), kernel=grad.transpose(1,0,2,3)).transpose(1,0,2,3)
        # in,N,H,W和 out,N,K,K 卷积完是 in,out,H',W'
        db = numpy.einsum('ijkl->j', out_grad)
        
        return Tensor(da), Tensor(dw), Tensor(db)
    
def conv2d(x, kernel, bias=None, stride=1, padding=0):
    # print(x.shape)
    return Conv2D(stride, padding)(x, kernel, bias)
    
class MaxPool2D(TensorOp):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def compute(self, x: NDArray) -> NDArray:
        N, C, H, W = x.shape
        H_out = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        W_out = (W - self.kernel_size + 2 * self.padding) // self.stride + 1
        x_padded = numpy.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        out = numpy.zeros((N, C, H_out, W_out), dtype=x.dtype)
        col = img2col(x_padded, self.kernel_size, self.stride, keep_channel=True)
        out = numpy.max(col, axis=1).reshape(N, C, H_out, W_out)   
        return out
        
    
    def gradient(self, out_grad, node):
        x = node.inputs[0]
        H_out, W_out = out_grad.shape[2], out_grad.shape[3]
        dx = numpy.zeros_like(x, dtype=x.dtype)
        for i in range(H_out):
            for j in range(W_out):
                x_slice = x[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                mask = (x_slice == numpy.max(x_slice, axis=(2, 3), keepdims=True))
                dx[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size] += out_grad[:, :, i, j][:, :, None, None] * mask
        
        return Tensor(dx)
    
def max_pool2d(x, kernel_size=2, stride=None, padding=0):
    return MaxPool2D(kernel_size, stride, padding)(x)
    
class AvgPool2D(TensorOp):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def compute(self, x: NDArray) -> NDArray:
        N, C, H, W = x.shape
        x_padded = numpy.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        H_out = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        W_out = (W - self.kernel_size + 2 * self.padding) // self.stride + 1
        out = numpy.zeros((N, C, H_out, W_out), dtype=x.dtype)
        
        # col = img2col(x, self.kernel_size, self.stride, keep_channel=True)
        # out = numpy.mean(col, axis=1).reshape(N, C, H_out, W_out)
        for i in range(H_out):
            for j in range(W_out):
                x_slice = x_padded[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                out[:, :, i, j] = numpy.mean(x_slice, axis=(2, 3))
        return out
    
    def gradient(self, out_grad, node):
        
        x = node.inputs[0].numpy()
        out_grad = out_grad.numpy()
        # print("AvgPool2D, out_grad shape", out_grad.shape, x.shape)
        H_out, W_out = out_grad.shape[2], out_grad.shape[3]
        dx = numpy.zeros(x.shape, dtype=x.dtype)
        for i in range(H_out):
            for j in range(W_out):
                dx[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size] += out_grad[:, :, i, j][:, :, numpy.newaxis, numpy.newaxis] / (self.kernel_size * self.kernel_size)
        return Tensor(dx)
    
def avg_pool2d(x,  kernel_size=2, stride=None, padding=0):
    return AvgPool2D(kernel_size, stride, padding)(x)