import numpy as np
import cv2
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'#指定使用的GPU设备号

from yolo import yolo_body
from yolo_layer import *
from utils.utils import *


class detect(object):
    def __init__(self,**kwargs):
        super(detect,self).__init__()

        #路径参数
        self.model_path = kwargs['model_path']#模型路径
        self.anchors_path = kwargs['anchors_path']#预设anchors路径
        self.classes_path = kwargs['classes_path']#类别名称路径
        #其他参数
        self.model_image_size = kwargs['model_image_size']
        self.confidence = kwargs['confidence']
        self.cuda = kwargs['cuda']
        #根据路径获得
        self.class_names = self.get_cls()
        self.anchors = self.get_anchors()
        self.net = yolo_body(3,len(self.class_names)).eval()#.eval()的意思是只有前馈，不计算梯度，在运行中提高速度
        self.load_model(self.net,self.model_path)

        if self.cuda:
            self.net = self.net.cuda()
            self.net.eval()

        #构造解码层
        self.yolo_decodes = []
        anchor_masks = [[1,2,3],[4,5,6],[7,8,9]]
        for i in range(3):
            head = YoloLayer(self.model_image_size,anchor_masks,len(self.class_names),
                                                self.anchors,len(self.anchors)//2).eval()
            self.yolo_decodes.append(head)

    def load_model(self,model,path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pth, map_location=device)
        matched_dict = {}
        #匹配检测
        for k, v in pretrained_dict.items():
            if np.shape(model_dict[k]) == np.shape(v):
                matched_dict[k] = v
            else:
                print('un matched layers: %s' % k)
        print(len(model_dict.keys()), len(pretrained_dict.keys()))
        print('%d layers matched,  %d layers miss' % (
        len(matched_dict.keys()), len(model_dict) - len(matched_dict.keys())))

        #更新/加载模型参数
        model_dict.update(matched_dict)
        model.load_state_dict(pretrained_dict)

        return model

    def get_cls(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]

        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]

        return anchors

    def detect_img(self,img_src):
        h,w,_ = img_src.shape
        img = cv2.resize(img_src,(608,608))#最近邻内插至所需要的shape
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        img = np.transpose(img/255.0,(2,0,1))
        images = np.asarray([img])

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        output_list = []
        for i in range(3):#共三个尺寸的anchors组和输出
            output_list.apend(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list,1)

        #非极大值抑制
        batch_detections = non_max_suppression(output,len(self.class_names),
                                                conf_thres=self.confidence,
                                                nms_thres=0.1)
        boxes  = [box.cpu().numpy() for box in batch_detections]

        print(boxes[0].shape)
        return boxes[0]

if __name__ == '__main__':
    params = {
        "model_path": 'pth/yolo4_weights_my.pth',
        "anchors_path": 'work_dir/yolo_anchors_coco.txt',
        "classes_path": 'work_dir/coco_classes.txt',
        "model_image_size": (608, 608, 3),
        "confidence": 0.4,
        "cuda": True
    }

    model = Inference(**params)
    class_names = load_class_names(params['classes_path'])
    image_src = cv2.imread('dog.jpg')
    boxes = model.detect_image(image_src)
    plot_boxes_cv2(image_src, boxes, savename='output3.jpg', class_names=class_names)