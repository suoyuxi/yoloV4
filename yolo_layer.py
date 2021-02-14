import torch.nn as nn
import torch

'''
decode函数和YOLOLAYER类都是对某个尺度的输出进行解码，比如out1或out2或out3
训练过程中会不经过解码直接把out输出与target进行loss计算，所以参与loss计算的数据没有经过sigmoid的0~1规整
'''

#将输出解码，形式上output[batch_size,A*n_channels,h,w]→[batch_size,A,h,w,n_channels]
def decode(output,num_classes,anchors,num_anchors,scale_xy):
	#判断数据位于显存还是内存，统一运行设备
	device = None
	if output.is_cuda:
		device = output.get_device()

	#获取每个anchors应该有的通道数
	n_channels = 4 + 1 + num_classes#x,y,w,h
	#获取anchors的数量，一般为3
	A = num_anchors
	#获取batch_size
	batch_size = output.size(0)
	#获取网格行数
	h = output.size(2)
	#获取网格列数
	w = output.size(3)

	#将output做shape转换
	#view和permute操作不会改变数据在内存的连续分布，导致无法再次对其进行shape变换，所以用contiguous进行内存改写
	output = output.view(batch_size,A,n_channels,h,w).permute(0,1,3,4,2).contiguous()
	
	#逐个获取预测的元素，使同属性量的可被同时操作解码
	bx,by = output[...,0],output[...,1]
	bw,bh = output[...,2],output[...,3]
	obj_confs,cls_confs = output[...,4],output[...,5:]

	#sigmoid激活，保证bx,by为0~1的比例性质的输出，obj_confs,cls_confs为0~1的置信度性质的输出
	bx = torch.sigmoid(bx)
	by = torch.sigmoid(by)
	obj_confs = torch.sigmoid(obj_confs)
	cls_confs = torch.sigmoid(cls_confs)

	#x,y加上格点坐标
	grid_x = torch.arange(w,dtype=torch.float).repeat(1,3,W,1).to(device)
	grid_y = torch.arange(h,dtype=torch.float).repeat(1,3,h,1).permute(0,1,3,2).to(device)

	bx += grid_x
	by += grid_y

	#针对每个anchors，根据bw，bh获得解码后的边框长宽
	bw = torch.exp(bw)*scale_x_y - 0.5*(scale_x_y-1)
	bh = torch.exp(bh)*scale_x_y - 0.5*(scale_x_y-1)

	for i in range(num_anchors):
		bw[:,i,:,:] *= anchors[i*2]
		bh[:,i,:,:] *= anchors[i*2+1]

	#获得相比较于整幅图像的0~1比例的bx,by,bw,bh
	bx = (bx/w).unsqueeze(-1)
	by = (by/h).unsqueeze(-1)
	bw = (bw/w).unsqueeze(-1)
	bh = (bh/h).unsqueeze(-1)

	#把这些解码好的数值特征重组成(B, A*H*W, num_classes)的形式
	boxes = torch.cat((bx, by, bw, bh), dim=-1).reshape(batch_size, A * h * w, 4)
	obj_confs = obj_confs.unsqueeze(-1).reshape(batch_size, A*h*w, 1)
	cls_confs =cls_confs.reshape(batch_size, A*h*w, num_classes)
	output = torch.cat([boxes, obj_confs, cls_confs], dim=-1)

	return output

class YoloLayer(nn.Module):
	def __init__(self,img_size,anchor_masks=[],num_classes=80,anchors=[],num_anchors=9,scale_x_y=1):
		super(YoloLayer, self).__init__()
		#[[6,7,8],[,3,4,5],[0,1,2]]
		self.anchor_masks = anchor_masks
		#类别
		self.num_classes = num_classes
		#anchors须是列表
		if type(anchors) == np.ndarray:
			self.anchors = anchors.tolist()
		else:
			self.anchors = anchors

		self.num_anchors = num_anchors
		self.anchor_step = len(self.anchors) // num_anchors

		self.scale_x_y = scale_x_y

		self.feature_length = [img_size[0]//8,img_size[0]//16,img_size[0]//32]
		self.img_size = img_size

	def forward(self, output):
		#训练情况下应直接返回网络输出，与target建立损失函数关系
		if self.training:
			return output

		in_w = output.size(3)
		anchor_index = self.anchor_masks[self.feature_length.index(in_w)]
		stride_w = self.img_size[0] / in_w
		masked_anchors = []
		for m in anchor_index:
			masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
		#每个anchors要除以单网格的边长
		self.masked_anchors = [anchor / stride_w for anchor in masked_anchors]

		data = yolo_decode(output, self.num_classes, self.masked_anchors, len(anchor_index),scale_x_y=self.scale_x_y)
		
		return data