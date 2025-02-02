import cv2
import numpy as np

class EmotionDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # 加载预训练模型
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def get_satisfaction_score(self, frames=5):
        """获取用户满意度分数 (0-1)"""
        total_score = 0
        valid_frames = 0
        
        for _ in range(frames):
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            try:
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 检测人脸
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # 简单的明亮度和对比度分析作为满意度指标
                    face_roi = gray[faces[0][1]:faces[0][1]+faces[0][3], 
                                  faces[0][0]:faces[0][0]+faces[0][2]]
                    brightness = np.mean(face_roi)
                    contrast = np.std(face_roi)
                    
                    # 将亮度和对比度映射到0-1的分数
                    score = min(1.0, (brightness/255.0 * 0.7 + contrast/128.0 * 0.3))
                    
                    total_score += score
                    valid_frames += 1
                
            except Exception as e:
                continue
        
        self.cap.release()
        return total_score / max(1, valid_frames) 