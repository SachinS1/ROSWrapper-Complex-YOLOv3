#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
import numpy as np
from cv_bridge import CvBridge
import cv2
import math
import os
import argparse
import time
import torch

import complexyolov3.utils.utils as utils
from complexyolov3.models import *
import torch.utils.data as torch_data

import complexyolov3.utils.kitti_utils as kitti_utils
import complexyolov3.utils.kitti_aug_utils as aug_utils
import complexyolov3.utils.kitti_bev_utils as bev_utils
from complexyolov3.utils.kitti_yolo_dataset import KittiYOLODataset
import complexyolov3.utils.config as cnf
import complexyolov3.utils.mayavi_viewer as mview
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
        

class LiDARNode:
    def __init__(self, point_cloud_topic, classes_file, config_file, model_checkpoint, bev_detection_out_topic, confidence_threshold, nms_threshold):
        
        self.pcl_subscriber = rospy.Subscriber(point_cloud_topic, PointCloud2, self.pcl_callback )
        self.classes = utils.load_classes(classes_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(config_file).to(self.device)
        self.model.load_state_dict(torch.load(model_checkpoint))
        self.model.eval()
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.bridge = CvBridge()
        self.img_publisher = rospy.Publisher(bev_detection_out_topic, Image, queue_size=10)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.first_scan = True
        rospy.spin()
    

    def pcl_callback(self, lidar_msg):
        self.lidar_msg = np.frombuffer(lidar_msg.data, np.float32).reshape(-1, int(lidar_msg.point_step/4))[:, :5]
        front_lidar = bev_utils.removePoints(self.lidar_msg, cnf.boundary)
        self.front_bev = bev_utils.makeBVFeature(front_lidar, cnf.DISCRETIZATION, cnf.boundary)
        
        self.front_bev = torch.tensor(self.front_bev).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).unsqueeze(0)
        with torch.no_grad():
            detections = self.model(self.front_bev)
            detections = utils.non_max_suppression_rotated_bbox(detections, 0.5, 0.5)
        
        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        bev_maps = torch.squeeze(self.front_bev).cpu()
        self.RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
        self.RGB_Map[:, :, 2] = bev_maps[0, :, :]  # r_map
        self.RGB_Map[:, :, 1] = bev_maps[1, :, :]  # g_map
        self.RGB_Map[:, :, 0] = bev_maps[2, :, :]  # b_map
        
        self.RGB_Map *= 255
        self.RGB_Map = self.RGB_Map.astype(np.uint8)
        
        for detections in img_detections:
            if detections is None:
                continue

            # Rescale boxes to original image
            detections = utils.rescale_boxes(detections, 608, self.RGB_Map.shape[:2])
            if self.first_scan:
                rospy.loginfo("Started BEV detection Node!")
                self.first_scan = False
            for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                bev_utils.drawRotatedBox(self.RGB_Map, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

        img_msg = self.bridge.cv2_to_imgmsg(self.RGB_Map, encoding="passthrough")

        self.img_publisher.publish(img_msg)
        


if __name__=="__main__":
    rospy.init_node("complexyolo_detector_node")
    point_cloud_topic = rospy.get_param("~point_cloud_topic")
    classes_file = rospy.get_param("~classes_file")
    config_file = rospy.get_param("~config_file")
    model_checkpoint = rospy.get_param("~model_checkpoint")
    bev_detection_out_topic = rospy.get_param("~bev_detection_out_topic")
    confidence_threshold = rospy.get_param("~confidence_threshold")
    nms_threshold = rospy.get_param("~nms_threshold")
    node = LiDARNode(point_cloud_topic, classes_file, config_file, model_checkpoint, bev_detection_out_topic, confidence_threshold, nms_threshold)
    cv2.destroyAllWindows()

    


