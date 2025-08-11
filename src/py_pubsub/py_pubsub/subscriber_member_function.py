"""
stop_sign_manager.py – Fuse camera & LiDAR, track vehicles with YOLO, and decide when it is safe to proceed at a stop‑sign‑controlled intersection.

Pipeline:
  1. Time‑synchronise Image + PointCloud2 + CameraInfo using message_filters.
  2. Project LiDAR points into the camera frame to obtain per‑pixel depth.
  3. Run Ultralytics YOLO on the RGB frame → bounding boxes, classes.
  4. Convert YOLO result to a list; pass to a centroid tracker that assigns stable IDs.
  5. Right‑of‑way manager keeps track of which tracked vehicles are still in the ROI and declares the scene clear when all have departed *and* LiDAR indicates no object is closer than the safety distance.
  6. Publish a Bool message `right_of_way/ego_safe`.

"""

import math
from typing import List, Tuple
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge


"""
copied over methods from stop_sign.py
"""
def get_detections_as_list(data):
   res = data
   bounding_box = res.boxes.xyxy.cpu().numpy().tolist() # (x1, y1, x2, y2)
   confs = res.boxes.conf.cpu().numpy().tolist()
   class_idx = res.boxes.cls.cpu().numpy().astype(int).tolist()

   names = []
   for c_idx in class_idx:
       names.append(res.names[c_idx])


   detections = []
   for box, conf, c_idx, label in zip(bounding_box, confs, class_idx, names):
       thing = [box[0], box[1], box[2], box[3], conf, c_idx, label]
       detections.append(thing)
   return detections

class Tracker:
   def __init__(self, max_distance=50):
       self.next_id = 0
       self.tracks = {}
       self.max_distance = max_distance
  
   def centroid(self, box):
       x1, y1, x2, y2 = box
       return ((x1+x2)/2, (y1+y2)/2)
  
   def update(self, detections):
       """
       detections: list of [x1, y1, x2, y2, conf, class_id, label]
       Returns a new list of
         [ (cx,cy), conf, class_id, label, track_id ]
       """
       assigned = []
       used_track_ids = set()

       detection_centroids = []
       for det in detections:
           box = det[0:4]
           centroid = self.centroid(box)
           detection_centroids.append(centroid)

       for det, centroid in zip(detections, detection_centroids):
           best_match_id = None
           best_match_dist = self.max_distance

           for track_id, track_centroid in self.tracks.items():
               if track_id in used_track_ids:
                   continue
              
               dx = centroid[0] - track_centroid[0]
               dy = centroid[1] - track_centroid[1]
               distance = math.hypot(dx, dy)

               if distance < best_match_dist:
                   best_match_dist = distance
                   best_match_id = track_id
          
           if best_match_id is not None:
               track_id = best_match_id
           else:
               track_id = self.next_id
               self.next_id += 1
          
           used_track_ids.add(track_id)
           self.tracks[track_id] = centroid

           conf = det[4]
           class_id = det[5]
           label = det[6]
           tracked_entry = [centroid, conf, class_id, label, track_id]
           assigned.append(tracked_entry)

       return assigned


class RightOfWayManager:
   def __init__(self, roi, max_missed_frames = 2):
       # roi = bounding box of the intersection area --> for now just whole screen
       self.roi = roi
       self.max_missed_frames = max_missed_frames
       self.seen_init_count = 0
       self.seen_initially = False
       self.priority_ids = set()
       self.priority_missed  = {}
       self.tracker = Tracker(max_distance=50)
  
   """
   checks if centroid still in ROI
   """
   def InROI(self, centroid):
       x, y = centroid
       x_min, y_min, x_max, y_max = self.roi
       return (x_min <= x <= x_max) and (y_min <= y <= y_max)
  
   def process_frame(self, res):
       detections = get_detections_as_list(res)
       tracked = self.tracker.update(detections)


       if not self.seen_initially:
           for entry in tracked:
               centroid, conf, class_id, label, track_id = entry
               if class_id in (2, 5, 7) and self.InROI(centroid):
                   self.priority_ids.add(track_id)
          
           for track_id in self.priority_ids:
               self.priority_missed[track_id] = 0
          
           if self.seen_init_count > 3:
               self.seen_initially = True
           else:
               self.seen_init_count+=1
               return False


       print("detections", tracked)


       still_in_ROI = set()


       # add all tracked entries to still_in_ROI set
       for entry in tracked:
           centroid, conf, class_id, label, track_id = entry
           if track_id in self.priority_ids and self.InROI(centroid):   
               still_in_ROI.add(track_id)
      
       # if expected vehicle was not detected in ROI, increment its missed counter in priority_missed
       for track_id in (self.priority_ids - still_in_ROI):
           self.priority_missed[track_id] = self.priority_missed[track_id] + 1
      
       # if expected vehicle was detected in ROI, set missed counter to 0 to reset the missed count
       for track_id in (still_in_ROI):
           self.priority_missed[track_id] = 0
      
       # if track_id in priority_ids has more than max_missed_frames missed frames ie yolo model hasn't seen it in a while,
       # assume it left the intersection, remove it from priority_ids and priority_missed
       for track_id in list(self.priority_ids):
           if self.priority_missed[track_id] > self.max_missed_frames:
               self.priority_ids.remove(track_id)
               del self.priority_missed[track_id]




       print("priority ids", self.priority_ids, "still in ROI", still_in_ROI, "missed counts", self.priority_missed)


       # once all priority_ids are gone, return true
       return len(self.priority_ids) == 0



# ROS2 node

class StopSignManager(Node):

    def __init__(self):
        super().__init__('stop_sign_manager')

        self.latest_image = None
        self.latest_lidar = None

        # params
        self.declare_parameter('sync.slop_ms', 50)
        self.declare_parameter('yolo.model', 'yolo11n.pt')
        self.declare_parameter('decision.safe_distance_m', 8.0)

        self.bridge = CvBridge()
        self.yolo = YOLO(self.get_parameter('yolo.model').value)
        INTERSECTION_ROI = (0, 0, 1920, 1080)  # Adjust to match your image resolution
        self.romgr = RightOfWayManager(roi=INTERSECTION_ROI, max_missed_frames=3)

        self.prev_sign_w = None
        self.prev_sign_h = None
        self.sign_stable_frames = 0
        self.stop_hold_frames = 3          # how many consecutive stable frames = "stopped"
        self.inc_percentage = 0.08           #  relative change counts as stable
        self.min_sign_conf = 0.2           # confidence threshold (matches your check)

        # subs
        custom_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        self.image_sub = self.create_subscription(Image, '/sensing/camera/camera0/image_rect_color', self.image_callback, qos_profile=custom_qos_profile)
        self.lidar_sub = self.create_subscription(PointCloud2, '/sensing/lidar/top/outlier_filtered/pointcloud', self.lidar_callback, qos_profile=custom_qos_profile)
        self._logged_enc = False
        self.get_logger().info('StopSignManager node started 🚗')

    def image_to_bgr(self, img_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


    def largest_stop_sign(self, detections):
        """Return (x1,y1,x2,y2,conf) for the largest-area stop sign above conf, else None."""
        best = None
        best_area = -1
        for d in detections:
            x1, y1, x2, y2, conf, cls, _ = d
            if cls == 11 and conf >= self.min_sign_conf:
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2, conf)
        return best


    def stopped(self, detections):
        """
        Return True if the chosen stop sign's apparent size is stable for stop_hold_frames.
        Stable = relative change of width/height < sign_size_eps.
        """
        ss = self.largest_stop_sign(detections)
        if ss is None:
            self.prev_sign_w = None
            self.prev_sign_h = None
            self.sign_stable_frames = 0
            return False

        x1, y1, x2, y2, conf = ss
        w = float(x2 - x1)
        h = float(y2 - y1)

        if self.prev_sign_w is None or self.prev_sign_h is None:
            # first sighting / after reset
            self.prev_sign_w, self.prev_sign_h = w, h
            self.sign_stable_frames = 0
            return False

        #  taking the change in width and height and dividing it by the previous width and height 
        # to see if it increased less than self.inc_percentage
        rel_dw = abs(w - self.prev_sign_w) / max(self.prev_sign_w, 1.0)
        rel_dh = abs(h - self.prev_sign_h) / max(self.prev_sign_h, 1.0)
        stable = (rel_dw < self.inc_percentage) and (rel_dh < self.inc_percentage)

        if stable:
            self.sign_stable_frames += 1
        else:
            self.sign_stable_frames = 0

        # Update baseline (EMA optional; simple assign is fine)
        self.prev_sign_w, self.prev_sign_h = w, h

        return self.sign_stable_frames >= self.stop_hold_frames


    def image_callback(self, msg):
        self.latest_image = msg
        frame = self.image_to_bgr(msg)

        # Run YOLO detection on the frame
        results = self.yolo(frame)[0]
        detections = get_detections_as_list(results)
        has_sign = self.largest_stop_sign(detections) is not None
        
        if has_sign:
            if self.stopped(detections):
                self.get_logger().info("STOP SIGN DETECTED! CHECKING INTERSECTION: ")
                can_go = self.romgr.process_frame(results)
                if can_go:
                    self.get_logger().info("SAFE TO GO")
                else:
                    self.get_logger().info("DO NOT GO")
            else:
                self.get_logger().info("SIGN PRESENT, NOT FULLY STOPPED YET")
        else:
            self.get_logger().info("NO STOP SIGN DETECTED, CONTINUING")


    def lidar_callback(self, msg):
        self.latest_lidar = msg
        self.get_logger().info(f'received lidar')
    



def main(args=None):
    rclpy.init(args=args)
    node = StopSignManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



