import torch
import torch.nn as nn

from CSPDarknet import darknet53

#基本的CBL模块 convolution + batchnormalization + leaky relu
class CBL(nn.Module):
	def __init__(self,filter_in,filter_out,kernel_size,stride=1):
		super(CBL, self).__init__()

		self.conv = nn.Conv2d(filter_in,filter_out,kernel_size,stride,kernel_size//2,bias=False)
		self.bn = nn.BatchNorm2d(filter_out)
		self.activation = nn.LeakyReLU(0.1)

	def forward(self,x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.activation(x)
		return x

#SPP模块，金字塔池化
class SPP(nn.Module):
	def __init__(self,pool_sizes = [5,9,13]):
		super(SPP,self).__init__()

		self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size,1,pool_size//2) for pool_size in pool_sizes])

	def forward(self,x):
		features = [maxpool(x) for maxpool in self.maxpools[::-1]]
		features = features + [x]
		features = torch.cat(features,dim = 1)

		return features

#Upsample 上采样模块，用于完成多尺度融合
class Upsample(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=1):
		'''
		根据YOLOv4的结构，这里的上采样模块和CBL模块写在一起
		上采样采用最近邻内插
		'''
		super(Upsample,self).__init__()

		self.upsample = nn.Sequential(
			CBL(in_channels,out_channels,kernel_size),
			nn.Upsample(scale_factor = 2,mode = 'nearest')
			)

	def forward(self,x):
		return self.upsample(x)

#定义Neck网络中的三层CBL，升维用3×3的卷积核，降维用1×1的卷积核，size_list = [512,1024]
def make_three_layers(size_list,in_channels):
	m = nn.Sequential(
		CBL(in_channels,size_list[0],1),
		CBL(size_list[0],size_list[1],3),
		CBL(size_list[1],size_list[0],1)
		)

	return m

#定义Neck网络中的五层CBL，升维用3×3的卷积核，降维用1×1的卷积核，size_list = [低维,高维]
def make_five_layers(size_list,in_channels):
	m = nn.Sequential(
		CBL(in_channels,size_list[0],1),
		CBL(size_list[0],size_list[1],3),
		CBL(size_list[1],size_list[0],1),
		CBL(size_list[0],size_list[1],3),
		CBL(size_list[1],size_list[0],1)
		)

	return m

#yolo_neck的导出网络部分，不是解码部分，由一个CBL和一个conv组成
def yolo_head(size_list,in_channels):
	m = nn.Sequential(
		CBL(in_channels,size_list[0],3),
		nn.Conv2d(size_list[0],size_list[1],1)
		)

	return m

#YOLOv4网络的整体，包含backbone neak head
class yolo_body(nn.Module):
	def __init__(self,num_anchors=3,num_classes=20):
		super(yolo_body,self).__init__()

		#backbone
		self.backbone = darknet53(None)

		#19×19特征图导出网络
		self.conv1 = make_three_layers([512,1024],1024)
		self.SPP = SPP()
		self.conv2 = make_three_layers([512,1024],2048)
		#19×19与38×38的融合部分网络
		self.upsample1 = Upsample(512,256)
		self.conv_up1 = CBL(512,256,1)
		self.five_convs1 = make_five_layers([256,512],512)
		#38×38和76×76的融合部分网络
		self.upsample2 = Upsample(256,128)
		self.conv_up2 = CBL(256,128,1)
		self.five_convs2 = make_five_layers([128,256],256)

		#输出头的输出通道数，自顶而下融合
		final_out_channels = num_anchors * (5 + num_classes)
		#76×76的输出头
		self.yolo_head1 = yolo_head([256,final_out_channels],128)
		#76×76下采样到38×38，并且通过五层CBL接到输出头
		self.downsample1 = CBL(128,256,3,stride = 2)#下采样率为2
		self.five_convs3 = make_five_layers([256,512],512)
		self.yolo_head2 = yolo_head([512,final_out_channels],256)
		#38×38下采样到19×19，并且通过五层CBL接到输出头
		self.downsample2 = CBL(256,512,3,stride = 2)#下采样率为2
		self.five_convs4 = make_five_layers([512,1024],1024)
		self.yolo_head3 = yolo_head([1024,final_out_channels],512)

	def forward(self,x):
		'''
		关于P1-P5五个part可以看结构图，有详细标注
		'''
		x2,x1,x0 = self.backbone(x)#x0是backbone的out3，x1是out2，x2是out1，由深到浅

		#==================================================
		#从backbone出来，自下而上的融合
		P5 = self.conv1(x0)
		P5 = self.SPP(P5)
		P5 = self.conv2(P5)

		#19×19和38×38融合
		P5_upsample = self.upsample1(P5)
		P4 = self.conv_up1(x1)
		P4 = torch.cat([P4,P5_upsample],dim = 1)
		P4 = self.five_convs1(P4)

		#38×38和76×76融合
		P4_upsample = self.upsample2(P4)
		P3 = self.conv_up2(x2)
		P3 = torch.cat([P3,P4_upsample],dim = 1)
		P3 = self.five_convs2(P3)
		#===================================================

		#---------------------------------------------------
		#融合处理后，自顶而下的融合
		#76×76下采样与38×38融合
		P3_downsample = self.downsample1(P3)
		P2 = torch.cat([P3_downsample,P4],dim=1)
		P2 = self.five_convs3(P2)

		#38×38下采样与19×19融合
		P4_downsample = self.downsample2(P4)
		P1 = torch.cat([P4_downsample,P5],dim=1)
		P1 = self.five_convs4(P1)
		#---------------------------------------------------

		#===================================================
		#过头输出
		#19×19输出
		out1 = self.yolo_head3(P1)
		#19×19输出
		out2 = self.yolo_head2(P2)
		#19×19输出
		out3 = self.yolo_head1(P3)

		return out1,out2,out3

if __name__ == '__main__':
	model = yolo_body(3,80)
	data_in = torch.randn(1,3,608,608)
	x,y,z = model(data_in)
	print(x.size())
	print(y.size())
	print(z.size())





