import torch
import torch.nn.functional as F
import torch.nn as nn

import math
import numpy as np

# Mish激活函数
class Mish(nn.Module):	
	def __init__(self):
		super(Mish,self).__init__()

	def forward(self,x):
		return x * torch.tanh(F.softplus(x))

#CBM:convolution + batch normalization + Mish 
class BasicConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1):
		super(BasicConv, self).__init__()

		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.activation = Mish()

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.activation(x)
		return x

#残差模块，用来构建CSP模块
class Resblock(nn.Module):
	def __init__(self,in_channels,hidden_channels=None):
		super(Resblock,self).__init__()

		if hidden_channels == None:
			hidden_channels = in_channels

		self.block = nn.Sequential(
			BasicConv(in_channels,hidden_channels,1),#1×1卷积
			BasicConv(hidden_channels,in_channels,3)#3×3卷积
			)

	def forward(self,x):
		return x + self.block(x)

#CSPX模块，X代表CSP模块里面residual block的数量，事实上是CBM+CSP+CBM
class CSPX(nn.Module):
	def __init__(self,in_channels,out_channels,X,first):
		super(CSPX,self).__init__()

		self.downsample = BasicConv(in_channels,out_channels,3,stride = 2)#只有下采样的CBM是3×3的卷积核

		if first:#网络的第一个CSPX具有结构上的特殊性,CBM不把channels减半，整合CBM输入通道为2*channels
			
			self.CBMright = BasicConv(out_channels, out_channels, 1)

			self.CBMin = BasicConv(out_channels, out_channels, 1)
			self.residual = Resblock(out_channels, hidden_channels=out_channels//2)
			self.CBMout  = BasicConv(out_channels,out_channels,1)
			
			self.concat_conv = BasicConv(out_channels*2, out_channels, 1)

		else:

			self.CBMright = BasicConv(out_channels, out_channels//2, 1)#通道减半，cat后channels不变
			
			self.CBMin = BasicConv(out_channels, out_channels//2, 1)
			self.residual = nn.Sequential(
				*[Resblock(out_channels//2) for _ in range(X)]
				#*[]的意思是把列表拆成元素，作为参数传入
			)
			self.CBMout = BasicConv(out_channels//2,out_channels//2,1)

			self.concat_conv = BasicConv(out_channels, out_channels, 1)

	def forward(self,x):

		x = self.downsample(x)#CBM下采样

		x_left = self.CBMin(x)
		x_left = self.residual(x_left)
		x_left = self.CBMout(x_left)#CSP的左边部分

		x_right = self.CBMright(x)#CSP右边部分

		x = torch.cat([x_left,x_right],dim = 1)#拼接左右输出
		x = self.concat_conv(x)

		return x 

#YOLOv4的backbonebufen CSPDarkNet
class CSPDarkNet(nn.Module):
	def __init__(self,Xs):#Xs就是一个X的列表[1,2,8,8,4]，表示各个CSPX模块的residual block有多少个
		super(CSPDarkNet,self).__init__()

		self.inplanes = 32#经过第一个CBM后，特征图的通道数目
		self.feature_channels = [64,128,256,512,1024]#每通过一个CSPX，特征图的channels变成两倍，尺寸缩减一半
			
		self.conv1 = BasicConv(3,self.inplanes,kernel_size=3,stride=1)#网络的第一个CBM模块，没有融入CSPX
		self.backbone = nn.ModuleList([
		CSPX(self.inplanes,self.feature_channels[0],Xs[0],first=True),
		CSPX(self.feature_channels[0],self.feature_channels[1],Xs[1],first=False),
		CSPX(self.feature_channels[1],self.feature_channels[2],Xs[2],first=False),
		CSPX(self.feature_channels[2],self.feature_channels[3],Xs[3],first=False),
		CSPX(self.feature_channels[3],self.feature_channels[4],Xs[4],first=False)
		])#既可以填充layer也可以填充nn.squentials

		# 进行权值初始化
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels#卷积核组矩阵元素总个数
				m.weight.data.normal_(0, math.sqrt(2. / n))#高斯分布初始化
			elif isinstance(    m, nn.BatchNorm2d):
				m.weight.data.fill_(1)#以1填充
				m.bias.data.zero_()#bias以0填充

	def forward(self,x):
		out0 = self.conv1(x)

		out1 = self.backbone[0](out0)
		out2 = self.backbone[1](out1)
		out3 = self.backbone[2](out2)
		out4 = self.backbone[3](out3)
		out5 = self.backbone[4](out4)

		return out3,out4,out5

#加载模型参数
def load_model_pth(model,pth):#能保证必然正确运不会因为不匹配而报错
	print('Loading weights into state dict, name: %s'%(pth))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#选择模型的设备cpu或gpu
	model_dict = model.state_dict()#模型参数整合为dict
	pretrained_dict = torch.load(pth, map_location=device)#加载训练好的模型参数（dict）
	
	#对比预训练权重是否和当前模型匹配
	matched_dict = {}
	for k,v in pretrained_dict.items():
		if np.shape(model_dict[k]) == np.shape(v):
			matched_dict[k] = v
		else:
			print('un matched layers: %s'%k)
	print(len(model_dict.keys()), len(pretrained_dict.keys()))
	print('%d layers matched,  %d layers miss'%(len(matched_dict.keys()), len(model_dict)-len(matched_dict.keys())))
	
	#更新匹配的权值
	model_dict.update(matched_dict)
	model.load_state_dict(model_dict)
	print('Finished!')

	return model

#生成backbone的darknet53，并且加载与训练参数
def darknet53(path_pretrained):
	model = CSPDarkNet([1,2,8,8,4])

	if path_pretrained:
		model = load_model_pth(model,path_pretrained)

	return model

if __name__ == '__main__':
	backbone = darknet53(False)
	
	for n,p in backbone.named_parameters():
		print(p.size())