import torch
import cv2
import numpy as np
from collections import deque
import math
import logging
from tkinter import messagebox
import tkinter as tk
import sys

class CircularMotionDetector:
    def __init__(self, buffer_size=128, min_radius=10, max_radius=300, debug=True):
        """
        Enhanced Circular Motion Detection with Alert System
        """
        # Minimal logging setup
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        
        # Initialize tkinter root for message box
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        
        # Counter for consecutive non-circular frames
        self.non_circular_count = 0
        self.max_non_circular_frames = 15  # Adjust this value to change sensitivity
        
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.conf = 0.1
            self.model.iou = 0.45
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
        
        self.pts = deque(maxlen=buffer_size)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.debug = debug
        
        self.debug_info = {
            'total_frames': 0,
            'points_tracked': [],
            'motion_analysis_history': []
        }
    
    def detect_circular_motion(self, frame):
        """
        Advanced circular motion detection with alert system
        """
        self.debug_info['total_frames'] += 1
        
        try:
            results = self.model(frame)
            
            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                self.pts.appendleft((center_x, center_y))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            self.debug_info['points_tracked'].append(list(self.pts))
            
            if len(self.pts) > 10:
                return self._detailed_motion_analysis(frame)
                
            self.non_circular_count += 1
            return {'detected': False, 'reason': 'Insufficient tracking points'}
        
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return {'detected': False, 'reason': 'Detection process failed'}
    
    def _detailed_motion_analysis(self, frame):
        try:
            points = np.array(self.pts)
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            
            radius = np.mean(distances)
            radius_std = np.std(distances)
            radius_variation = radius_std / radius
            
            is_circular = (
                self.min_radius < radius < self.max_radius and
                radius_variation < 0.3
            )
            
            motion_analysis = {
                'is_circular': is_circular,
                'center_x': centroid[0],
                'center_y': centroid[1],
                'radius': radius,
                'radius_std': radius_std,
                'radius_variation': radius_variation
            }
            
            self.debug_info['motion_analysis_history'].append(motion_analysis)
            
            if is_circular:
                self.non_circular_count = 0  # Reset counter when circular motion is detected
                cv2.circle(frame, 
                           (int(centroid[0]), int(centroid[1])), 
                           int(radius), 
                           (0, 255, 0), 2)
                
                return {
                    'detected': True,
                    'is_circular': True,
                    'details': motion_analysis
                }
            
            self.non_circular_count += 1
            return {
                'detected': True,
                'is_circular': False,
                'details': motion_analysis
            }
        
        except Exception as e:
            self.logger.error(f"Motion analysis error: {e}")
            return {'detected': False, 'reason': 'Motion analysis failed'}
    
    def check_no_circular_motion(self):
        """
        Check if there's been no circular motion for too long
        """
        return self.non_circular_count >= self.max_non_circular_frames
    
    def show_alert(self):
        """
        Show alert message and return True if user wants to exit
        """
        response = messagebox.showwarning("No Circular Motion", 
                                        "FAULT DETECTED. Motion stopped.", 
                                        icon='warning')
        return True
    
    def generate_debug_report(self):
        """
        Generate minimal debug report
        """
        report = f"""=== Circular Motion Detection Debug Report ===
        Total Frames Processed: {self.debug_info['total_frames']}
        
        Tracking Points:
        - Number of tracking sequences: {len(self.debug_info['points_tracked'])}
        - Last tracking sequence length: {len(self.debug_info['points_tracked'][-1]) if self.debug_info['points_tracked'] else 0}"""
        return report

def main():
    video_path = r"C:\Users\Pragnesh Ghoniya\Downloads\trial (1).mp4"
    
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    detector = CircularMotionDetector(debug=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 360))
        result = detector.detect_circular_motion(frame)
        cv2.imshow('Circular Motion Detection', frame)
        
        # Check for no circular motion
        if detector.check_no_circular_motion():
            if detector.show_alert():
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(detector.generate_debug_report())
    
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == '__main__':
    main()