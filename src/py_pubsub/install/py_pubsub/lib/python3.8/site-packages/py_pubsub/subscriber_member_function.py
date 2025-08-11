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
from rclpy.duration import Duration

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Bool

from message_filters import Subscriber as MFSubscriber, ApproximateTimeSynchronizer

from sensor_msgs_py import point_cloud2      # PointCloud2 → NumPy iterator
from cv_bridge import CvBridge               # ROS Image ↔︎ OpenCV
import tf2_ros                               # TF buffer / listener


# helper to convert TF TransformStamped → 4×4 matrix without KDL
from tf_transformations import quaternion_matrix

def tf_to_matrix(ts: tf2_ros.TransformStamped) -> np.ndarray:
    q = [ts.transform.rotation.x, ts.transform.rotation.y,
         ts.transform.rotation.z, ts.transform.rotation.w]
    T = quaternion_matrix(q)
    T[0, 3] = ts.transform.translation.x
    T[1, 3] = ts.transform.translation.y
    T[2, 3] = ts.transform.translation.z
    return T  # 4×4 float64

"""
copied over methods from stop_sign.py
"""

def get_detections_as_list(res) -> List[List]:
    """
    Convert a Ultralytics result object to a list of detections.
    Each entry: [x1, y1, x2, y2, conf, class_id, label]
    """
    boxes = res.boxes
    bounding_box = boxes.xyxy.cpu().numpy().tolist()
    confs = boxes.conf.cpu().numpy().tolist()
    class_idx = boxes.cls.cpu().numpy().astype(int).tolist()
    names = [res.names[c] for c in class_idx]
    return [[*box, conf, cid, name] for box, conf, cid, name in zip(bounding_box, confs, class_idx, names)]


class Tracker:
    """
    Very small centroid tracker (no Kalman filter)
    """

    def __init__(self, max_distance: float = 50.0):
        self.next_id = 0
        self.tracks = {}  # dictionary mapping track_id --> (cx, cy)
        self.max_distance = max_distance

    @staticmethod
    def _centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections: List[List]):
        assigned = []
        used_tracks = set()
        for det in detections:
            c = self._centroid(det[:4])
            best_id, best_dist = None, self.max_distance
            for tid, t_c in self.tracks.items():
                if tid in used_tracks:
                    continue
                dist = math.hypot(c[0] - t_c[0], c[1] - t_c[1])
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            self.tracks[best_id] = c
            used_tracks.add(best_id)
            assigned.append([c, *det[4:], best_id])  # [centroid, conf, class_id, label, track_id]
        return assigned


class RightOfWayManager:
    """
    Implements simple 4‑way stop priority logic.

    – During an initial window, note every vehicle (class id 2,5,7) that appears in ROI → `priority_ids`.
    – Once the window closes, wait until all priority_ids have left the ROI for ≥ `max_missed_frames` frames.
    – When none remain, return True (safe to proceed).
    """

    VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck in COCO

    def __init__(self, roi: Tuple[int, int, int, int], first_window_frames: int = 10, max_missed_frames: int = 8):
        self.roi = roi
        self.max_missed = max_missed_frames
        self.first_window_frames = first_window_frames
        self.tracker = Tracker(max_distance=50)

        self._frame_count = 0
        self.priority_ids = set()
        self.missed = {}

    def _in_roi(self, cx, cy):
        x_min, y_min, x_max, y_max = self.roi
        return x_min <= cx <= x_max and y_min <= cy <= y_max

    def process_yolo(self, res):
        # Update tracker and obtain per‑detection info with ID
        detections = get_detections_as_list(res)
        tracked = self.tracker.update(detections)

        # Phase 1: gather initial set of vehicles
        if self._frame_count < self.first_window_frames:
            for (cx, cy), conf, cid, label, tid in tracked:
                if cid in self.VEHICLE_CLASS_IDS and self._in_roi(cx, cy):
                    self.priority_ids.add(tid)
                    self.missed[tid] = 0
            self._frame_count += 1
            return False  # not ready to decide yet

        # Phase 2: monitor departure
        still_here = set()
        for (cx, cy), conf, cid, label, tid in tracked:
            if tid in self.priority_ids and self._in_roi(cx, cy):
                still_here.add(tid)

        for tid in self.priority_ids:
            if tid in still_here:
                self.missed[tid] = 0
            else:
                self.missed[tid] = self.missed.get(tid, 0) + 1
        # Remove those that have been gone for long enough
        for tid in list(self.priority_ids):
            if self.missed.get(tid, 0) > self.max_missed:
                self.priority_ids.discard(tid)
                self.missed.pop(tid, None)

        return len(self.priority_ids) == 0



# ============================================================
#  ROS2 node that fuses LiDAR + image and uses RightOfWayManager
# ============================================================

class StopSignManager(Node):

    def __init__(self):
        super().__init__('stop_sign_manager')

        # params
        self.declare_parameter('sync.slop_ms', 50)
        self.declare_parameter('yolo.model', 'yolo11n.pt')
        self.declare_parameter('decision.safe_distance_m', 8.0)

        # self.declare_parameter('camera.fx', 600.0)    # fill with actual fx
        # self.declare_parameter('camera.fy', 600.0)
        # self.declare_parameter('camera.cx', 320.0)
        # self.declare_parameter('camera.cy', 240.0)

        # subs
        custom_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        # self.image_sub = self.create_subscription(Image, '/sensing/camera/camera0/image_rect_color', self.subscriber_callback, custom_qos_profile)
        self.image_sub = MFSubscriber(self, Image, '/sensing/camera/camera0/image_rect_color', qos_profile=custom_qos_profile)
        self.lidar_sub = MFSubscriber(self, PointCloud2, '/sensing/lidar/top/outlier_filtered/pointcloud', qos_profile=custom_qos_profile)
        # self.info_sub = MFSubscriber(self, CameraInfo, '/sensing/camera/camera0/camera_info')

        slop = self.get_parameter('sync.slop_ms').value / 1000.0
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], queue_size=10, slop=slop)
        self.ts.registerCallback(self._synced_cb)

        # tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # perception
        self.yolo = YOLO(self.get_parameter('yolo.model').value)
        self.bridge = CvBridge()

        # publish
        self.decision_pub = self.create_publisher(Bool, 'right_of_way/ego_safe', 1)

        # right of way manager
        self.romgr = None

        self.get_logger().info('StopSignManager node started 🚗')
    
    def subscriber_callback(self, msg):
        self.get_logger().info(f'Received message')

    # ---------------------------------------------------------
    #  Main callback with time‑aligned (image, cloud, info)
    # ---------------------------------------------------------
    def _synced_cb(self, img_msg: Image, pc_msg: PointCloud2):
        self.get_logger().info(f'Received message')
        # convert image to OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        # Build ROI for manager on first frame (full image)
        if self.romgr is None:
            self.romgr = RightOfWayManager(roi=(0, 0, w, h))

        # convert cloud to NumPy
        pts = np.array(list(point_cloud2.read_points(pc_msg, field_names=('x', 'y', 'z'), skip_nans=True)), dtype=np.float32)
        if pts.size == 0:
            self.get_logger().warn('Empty point cloud – skipping frame')
            return

        
        # transform cloud into camera frame via TF
        try:
            ts = self.tf_buffer.lookup_transform(
                'camera0_link', pc_msg.header.frame_id, pc_msg.header.stamp,
                Duration(seconds=0.05)
            )
            T = tf_to_matrix(ts).astype(np.float32)
            pts_cam = (T @ np.hstack([pts, np.ones((pts.shape[0],1))]).T).T[:,:3]
        except Exception as e:
            self.get_logger().warn(f'TF error: {e}'); return
        # LiDAR safety bubble
        front = pts_cam[pts_cam[:,2]>0.1]
        min_z = np.inf if front.size==0 else float(front[:,2].min())
        safe_dist = self.get_parameter('decision.safe_distance_m').value
        lidar_clear = min_z > safe_dist

        # yolo stuff
        yolo_res = self.yolo(frame, verbose=False, imgsz=640)[0]
        romgr_clear = self.romgr.process_yolo(yolo_res)
        safe = lidar_clear and romgr_clear
        self.decision_pub.publish(Bool(data=safe))
        self.get_logger().info(f'SAFE={safe}  (LiDAR {lidar_clear}, RoW {romgr_clear}, closest={min_z:.2f} m)')

        # Optional visual debug (comment out in headless mode)
        # cv2.imshow('frame', frame); cv2.waitKey(1)


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



















"""
hard reset


class StopSignManager(Node):

    def __init__(self):
        super().__init__('stop_sign_manager')
        self.image_sub = self.create_subscription(
            Image,
            '/sensing/camera/camera0/image_rect_color',
            self.image_callback,
            10)
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/sensing/lidar/top/outlier_filtered/pointcloud',
            self.lidar_callback,
            10)

    def image_callback(self, msg):
        self.get_logger().info('I heard:')

    def lidar_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = StopSignManager()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

"""
