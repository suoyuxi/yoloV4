import torch.nn as nn
import torch

from utils.utils import bbox_iou,merge_bboxes

def iou(_box_a,_box_b):
	'''
	注意两个box组的box个数未必一致
	_box_a: A * 4
	_box_b: B * 4
	'''
	b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
	b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
	b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
	b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

	box_a = torch.zeros_like(_box_a)
	box_b = torch.zeros_like(_box_b)

	box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
	box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

	A = box_a.size(0)
	B = box_b.size(0)

	max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
						box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
	min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
						box_b[:, :2].unsqueeze(0).expand(A, B, 2))
	inter = torch.clamp((max_xy - min_xy), min=0)

	inter = inter[:, :, 0] * inter[:, :, 1]
	# 计算先验框和真实框各自的面积
	area_a = ((box_a[:, 2]-box_a[:, 0]) *
				(box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
	area_b = ((box_b[:, 2]-box_b[:, 0]) *
				(box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
	# 求IOU
	union = area_a + area_b - inter
	return inter / union  # [A,B]

def smooth_labels(y_true,label_smoothing,num_classes):
	return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes#提高后验熵，hinton这样做过，也是知识蒸馏的早期灵感

def box_ciou(b1,b2):
	#b1,b2: batch_size * 4

	#预测框
	b1_xy = b1[...,:2]
	b1_wh = b1[...,2:4]
	b1_wh_half = b1_wh / 2
	b1_min = b1_xy - b1_wh_half #左上角坐标
	b1_max = b1_xy + b1_wh_half#右上角坐标
	#真实框
	b2_xy = b2[...,:2]
	b2_wh = b2[...,2:4]
	b2_wh_half = b2_wh / 2
	b2_min = b2_xy - b2_wh_half#左上角坐标
	b2_max = b2_xy + b2_wh_half#右上角坐标

	#计算iou
	#相交部分的面积
	intersect_min = torch.max(b1_min,b2_min)#交叠部分的左上角坐标
	intersect_max = torch.min(b1_max,b2_max)#交叠部分的右上角坐标
	intersect_wh = torch.max((intersect_max - intersect_min),torch.zeros_like(intersect_max))

	area_intersect = intersect_wh[...,0] * intersect_wh[...,1]
	area_b1 = b1_wh[...,0] * b1_wh[...,1]
	area_b2 = b2_wh[...,0] * b2_wh[...,1]
	area_uion = area_b1 + area_b2 - area_intersect

	iou = area_intersect / torch.clamp(area_uion, min=1e-6)

	#计算d^2/c^2
	center_distance_power_2 = torch.sum(torch.pow((b1_xy - b2_xy),2), axis= -1)#d^2

	enclose_min = torch.min(b1_min,b2_min)
	enclose_max = torch.max(b1_max,b2_max)
	enclose_distance_power_2 = torch.sum(torch.pow((enclose_max - enclose_min),2), axis= -1)#c^2

	d2c2 = center_distance_power_2 / torch.clamp(enclose_distance_power_2, min=1e-6)

	diou = iou - 1.0 * d2c2

	#alpha & v
	v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
	alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
	ciou = ciou - alpha * v

	return ciou





#把tensor截断至min到max之间
def clip_by_tensor(t,t_min,t_max):
	t=t.float()
	result = (t >= t_min).float() * t + (t < t_min).float() * t_min
	result = (result <= t_max).float() * result + (result > t_max).float() * t_max
	
	return result

def MSELoss(pred,target):
	return (pred - target)**2

#二分类交叉熵
def BCELoss(pred,target):
	epsilon = 1e-7
	pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
	output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
	
	return output

class YOLOLoss(nn.Module):
	def __init__(self,anchors,num_classes,img_size,label_smooth=0,cuda=True):
		super(YOLOLoss,self).__init__()

		self.anchors = anchors
		self.num_anchors = len(anchors)
		self.num_classes = num_classes
		self.bbox_attrs = 5 + num_classes#每个格应包含的属性数目
		self.img_size = img_size#网络输入的尺寸（608×608）
		self.features_length = [img_size[0]//8,img_size[0]//16,img_size[0]//32]#三种输出特征图的尺寸，或每行/列的格数，8、16、32是每个格子的像素数
		self.label_smooth = label_smooth		

		self.ignore_threshold = 0.7#负样本筛选阈值
		#各类损失的权衡系数
		self.lambda_conf = 1.0#obj置信损失
		self.lambda_cls = 1.0#分类损失
		self.lambda_loc = 1.0#框的损失
		self.cuda = cuda

	def forward(self,input,target):
		#input:batch_size×(3*(5+num_classes))×H×W

		batch_size = input.size(0)
		in_h = input.size(2)
		in_w = input.size(2)

		stride_h = self.img_size[1] / in_h
		stride_w = self.img_size[0] / in_w#每个格子的像素长宽8，16，32

		#将先验框anchors变为特征图上的尺寸，即除以格的像素宽高
		scaled_anchors = [(a_w/stride_w,a_h/stride_h) for a_w,a_h in self.anchors]
		#把三个anchors在input中对应的部分拆开→batch_size*num_anchors*h*w*bbox_attrs
		prediction = input.view(batch_size,int(self.num_anchors/3),self.bbox_attrs,in_h,in_w).permute(0,1,3,4,2).contiguous()

		#对prediction进行0~1化调整，注意，训练过程中的网络输出是不经过yolo_layer解码的
		conf = torch.sigmoid(prediction[...,4])#obj置信度
		pred_cls = torch.sigmoid(prediction[...,5:])#类别置信度（后验概率）

		#找到那些先验框内部包含物体
		mask,noobj_mask,t_box,tconf,tcls,box_loss_scale_x,box_loss_scale_y = self.get_target(targets,scaled_anchors,in_w,in_h,self.ignore_threshold)

		noobj_mask, pred_boxes_for_iou = self.get_ignore(prediction,targets,scaled_anchors,in_w,in_h,noobj_mask)

		if self.cuda:
			mask,noobj_mask = mask.cuda(),noobj_mask.cuda
			box_loss_scale_x,box_loss_scale_y = box_loss_scale_x.cuda(),box_loss_scale_y.cuda()
			tconf,tcls = tconf.cuda(),tcls.cuda()
			pred_boxes_for_iou = pred_boxes_for_iou.cuda()
			t_box = t_box.cuda()

		box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y

		#计算各项损失
		ciou = box_ciou(pred_boxes_for_iou[mask.bool()],t_box[mask.bool()])
		loss_ciou = 1 - ciou
		loss_ciou = loss_ciou * box_loss_scale[mask.bool()]#目标检测尤其是yolo对小目标不友好，因此给小目标损失加高权
		loss_loc = torch.sum(loss_ciou/batch_size)

		loss_conf = torch.sum(BCELoss(conf,mask) * mask / batch_size) + \
					torch.sum(BCELoss(conf,mask) * noobj_mask / batch_size)

		loss_cls = torch.sum(BCELoss(pred_cls[mask == 1],smooth_labels(tcls[mask == 1],self.label_smooth,self.num_classes))/batch_size)

		loss = loss_cls * self.lambda_cls + loss_conf * self.lambda_conf + loss_loc * self.lambda_loc

		return loss,loss_conf.item(),loss_cls.item(),loss_loc.item()

	def get_target(self,target,anchors,in_w,in_h,ignore_threshold):
		batch_size = len(target)#target是由dataloader给到的一个batch_size的groud_truth

		#取出先验框
		anchor_index = [[0,1,2][3,4,5],[6,7,8]][self.features_length.index(in_h)]#anchors中的绝对索引
		subtract_index = [0,3,6][self.features_length.index(in_h)]#绝对索引的offset，为了获得相对索引

		#初始化掩模
		mask = torch.zeros(batch_size,int(self.num_anchors/3),in_h,in_w,requires_grad = False)#requirs_grad = False是为了节省显存
		noobj_mask = torch.ones(batch_size,int(self.num_anchors/3),in_h,in_w,requires_grad = False)
		#初始化target
		tx = torch.zeros(batch_size,int(self.num_anchors/3),in_h,in_w,requires_grad = False)
		ty = torch.zeros(batch_size,int(self.num_anchors/3),in_h,in_w,requires_grad = False)
		tw = torch.zeros(batch_size,int(self.num_anchors/3),in_h,in_w,requires_grad = False)
		th = torch.zeros(batch_size,int(self.num_anchors/3),in_h,in_w,requires_grad = False)
		t_box = torch.zeros(batch_size,int(self.num_anchors/3),in_h,in_w,4,requires_grad = False)
		tconf = torch.zeros(batch_size,int(self.num_anchors/3),in_h,in_w,requires_grad = False)
		tcls = torch.zeros(batch_size,int(self.num_anchors/3),in_h,in_w,self.num_classes,requires_grad = False)

		box_loss_scale_x = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
		box_loss_scale_y = torch.zeros(batch_size, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        
        #target的格式是batch_size×num_objs×[x,y,w,h,cls]
		for b in range(batch_size):
			for t in range(target[b].shape[0]):#t表示第b张图片的第t个目标物体
				#target的x,y,w,h是相对于图片的0·1相对位置，要换算成特征图点位
				gx = target[b][t,0] * in_w
				gy = target[b][t,1] * in_h
				gw = target[b][t,2] * in_w
				gh = target[b][t,3] * in_h

				#计算出网格的二维索引
				gi = int(gx)
				gj = int(gy)

				#以真实框中心为原点，给出真实框的唯一表示[x,y,w,h]
				gt_box = torch.FloatTensor(np.array([0,0,gw,gh])).unsqueeze(0)

				#在此二维坐标系下给出所有先验框的唯一表示
				anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors,2)),np.array(anchors)),1))

				#计算每个anchor和真实框的iou，以方便选出是哪一个anchor用来表示真实框
				anch_ious = bbox_iou(gt_box,anchor_shapes)

				#找到这个最佳anchor
				best_n = np.argmax(anch_ious)
				if best_n not in anchor_index:
					continue#本目标不计入当前尺寸的target
				
				#填充赋值
				if (gj < in_h) and (gi < in_w):
					best_n = best_n - subtract_index#获取相对索引

					noobj_mask[b,best_n,gj,gi] = 0
					mask[b,best_n,gj,gi] = 1
					tx[b,best_n,gj,gi] = gx
					ty[b,best_n,gj,gi] = gy
					tw[b,best_n,gj,gi] = gw
					th[b,best_n,gj,gi] = gh
					tconf[b,best_n,gj,gi] = 1
					tcls[b,best_n,gj,gi,int(target[b][t,4])] = 1

					# 用于获得xywh的比例
					box_loss_scale_x[b, best_n, gj, gi] = target[b][t, 2]
					box_loss_scale_y[b, best_n, gj, gi] = target[b][t, 3]

				else:
					print('Step {0} out of bound'.format(b))
					print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
					continue

		#此时的t_box是相对特征图的尺寸，即取值范围0~in_W,0~in_H
		t_box[...,0] = tx
		t_box[...,1] = ty
		t_box[...,2] = tw
		t_box[...,3] = th

		return mask,noobj_mask,t_box,tconf,tcls,box_loss_scale_x,box_loss_scale_y



	def get_ignore(self,prediction,target,scaled_anchors,in_w, in_h,noobj_mask):
		batch_size = len(target)
		anchor_index = [[0,1,2],[3,4,5],[6,7,8]][self.feature_length.index(in_w)]
		scaled_anchors = np.array(scaled_anchors)[anchor_index]#是除以每个格子的像素宽高后的特征图anchors
		# 先验框的中心位置的调整参数
		x = torch.sigmoid(prediction[..., 0])  
		y = torch.sigmoid(prediction[..., 1])
		# 先验框的宽高调整参数
		w = prediction[..., 2]  # Width
		h = prediction[..., 3]  # Height

		FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
		LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

		# 生成网格，先验框中心，网格左上角
		grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
			int(batch_size*self.num_anchors/3), 1, 1).view(x.shape).type(FloatTensor)
		grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
			int(batch_size*self.num_anchors/3), 1, 1).view(y.shape).type(FloatTensor)

		# 生成先验框的宽高
		anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
		anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
		
		anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(w.shape)
		anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(h.shape)
		
		# 计算调整后的先验框中心与宽高
		pred_boxes = FloatTensor(prediction[..., :4].shape)
		pred_boxes[..., 0] = x + grid_x
		pred_boxes[..., 1] = y + grid_y
		pred_boxes[..., 2] = torch.exp(w) * anchor_w
		pred_boxes[..., 3] = torch.exp(h) * anchor_h

		for i in range(batch_size):
			pred_boxes_for_ignore = pred_boxes[i]
			pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
			if len(target[i]) > 0:
				gx = target[i][:, 0:1] * in_w
				gy = target[i][:, 1:2] * in_h
				gw = target[i][:, 2:3] * in_w
				gh = target[i][:, 3:4] * in_h
				gt_box = torch.FloatTensor(np.concatenate([gx, gy, gw, gh],-1)).type(FloatTensor)

				anch_ious = iou(gt_box, pred_boxes_for_ignore)
				for t in range(target[i].shape[0]):
					anch_iou = anch_ious[t].view(pred_boxes[i].size()[:3])
					noobj_mask[i][anch_iou>self.ignore_threshold] = 0#pred_box和target_box的中心格点可能偏开了，比如pred的中心为（7，8）而真实为（7，7），但就算这样他们的iou依旧可能大于0.7，（7，8）要算成noobj=0
																		#也就是说真实的目标中心格点和iou大于0.7的格点都要算成noobj=0，noobj跟obj其掩模不是简单的取反
																		#如此计算noobj_mask的原因在于这部分的框不能单纯当作负样本，因此作为不计算的部分，即样本由正样本、负样本、不参与计算样本构成
																		#不参与计算样本据不能当作正样本，因为没有对应的bbox和cls标签
																		#但是如果把不参与计算样本当成负样本就会抑制网络向正确结果的靠拢收敛，所以索性不使其参与计算
		return noobj_mask, pred_boxes
		#网络输出是未经sigmoid的x,y,obj_conf,cls_conf,和为exp(·)*anchors的w,h，
		#而且x,y,w,h都是相对于特征图来说的即0~19或0~38或0~76,用于计算loss的target和pred的box都是这个尺寸，解码成这个尺寸常规的格式后计算损失
