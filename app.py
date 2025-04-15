import sys
import cv2
import numpy as np
import time
import pyttsx3
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
from cvzone.HandTrackingModule import HandDetector
import os
import re
import nltk
from nltk.corpus import words as nltk_words
from collections import defaultdict
import mediapipe as mp
import datetime
import json
import hashlib
import base64
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QGroupBox, QTextEdit, QCheckBox, QFrame,
                            QSplitter, QSizePolicy, QScrollArea, QTabWidget, QMessageBox, QDialog)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal, QSize
import math

# Download NLTK data if not already downloaded
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

# Offset for hand bounding box
offset = 20

# Global icon path for consistent usage throughout the app
ICON_PATH = os.path.join(os.path.dirname(__file__), 'icon.png')

class LicenseManager:
    """Manages application licensing and trial period"""
    LICENSE_FILE = os.path.join(os.path.expanduser('~'), '.sign_language_app_license.json')
    TRIAL_DAYS = 30
    # Hidden watermark information
    __AUTHOR = "Created by Adesh Patil"
    
    def __init__(self):
        self.first_run_date = None
        self.license_key = None
        self.load_license()
    
    def load_license(self):
        """Load license information from file"""
        if os.path.exists(self.LICENSE_FILE):
            try:
                with open(self.LICENSE_FILE, 'r') as f:
                    data = json.load(f)
                    self.first_run_date = data.get('first_run_date')
                    self.license_key = data.get('license_key')
            except Exception as e:
                print(f"Error loading license: {str(e)}")
        else:
            # First-time run, create license file
            self.first_run_date = datetime.datetime.now().strftime('%Y-%m-%d')
            self.save_license()
    
    def save_license(self):
        """Save license information to file"""
        try:
            data = {
                'first_run_date': self.first_run_date,
                'license_key': self.license_key,
                # Add hidden metadata with author info
                'metadata': self._encode_metadata()
            }
            with open(self.LICENSE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving license: {str(e)}")
    
    def _encode_metadata(self):
        """Encode hidden metadata"""
        # This encodes the author information but isn't directly visible
        metadata = {'author': self.__AUTHOR, 'timestamp': time.time()}
        return base64.b64encode(json.dumps(metadata).encode()).decode()
    
    def validate_license(self):
        """Validate the license and check trial period"""
        # If there's a valid license key, the app is fully licensed
        if self.license_key and self._validate_key(self.license_key):
            return True, None
        
        # Otherwise, check trial period
        if self.first_run_date:
            try:
                first_date = datetime.datetime.strptime(self.first_run_date, '%Y-%m-%d')
                current_date = datetime.datetime.now()
                days_passed = (current_date - first_date).days
                
                if days_passed <= self.TRIAL_DAYS:
                    days_left = self.TRIAL_DAYS - days_passed
                    return True, f"Trial version: {days_left} days remaining"
                else:
                    return False, "Trial period has expired. Please purchase a license."
            except Exception as e:
                print(f"Error validating license: {str(e)}")
                return False, "License validation error. Please reinstall the application."
        
        return False, "License information missing. Please reinstall the application."
    
    def _validate_key(self, key):
        """Validate a license key"""
        # This would normally validate against a server or use cryptographic verification
        # For this example, we'll just do a simple check
        if key and len(key) == 32:
            return True
        return False
    
    def register_license(self, key):
        """Register a new license key"""
        if self._validate_key(key):
            self.license_key = key
            self.save_license()
            return True
        return False
    
    def get_watermark(self):
        """Get the hidden watermark text - for internal use only"""
        return self.__AUTHOR

class VideoThread(QThread):
    """Thread for processing video frames"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    prediction_signal = pyqtSignal(str)
    word_signal = pyqtSignal(str)
    suggestions_signal = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        
        # Add debug_mode flag
        self.debug_mode = False
        
        # Load the TensorFlow model
        try:
            self.model = tf.keras.models.load_model('cnn8grps_rad1_model.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands with optimized parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # For hand detection
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # For hand tracking
        self.hand_detector2 = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Variables for tracking predictions
        self.current_symbol = None
        self.prev_symbol = None
        self.count_same_symbol = 0
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.5  # seconds
        
        # Frame processing optimization
        self.frame_skip = 1  # Process every other frame
        self.frame_count = 0
        self.last_processed_frame = None
        
        # Prediction confidence threshold
        self.confidence_threshold = 0.65
        
        # Prediction history for smoothing
        self.prediction_history = []
        self.history_size = 5
        
        # Text output tracking
        self.current_word = ""
        self.output_text = ""
    
    def run(self):
        """Main thread function"""
        # Open the camera
        cap = cv2.VideoCapture(0)
        
        # Set lower resolution for faster processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames for better performance
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                if self.last_processed_frame is not None:
                    self.change_pixmap_signal.emit(self.last_processed_frame)
                continue
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            self.last_processed_frame = processed_frame
            
            # Emit the processed frame
            self.change_pixmap_signal.emit(processed_frame)
        
        # Release the camera
        cap.release()
    
    def process_frame(self, frame):
        """Process a video frame for hand detection"""
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Make a copy of the original frame to preserve its color
        original_frame = frame.copy()
        
        # Add debug info if debug mode is enabled
        if self.debug_mode:
            # Get frame width and height for positioning
            h, w = original_frame.shape[:2]
            
            # Add frame processing info - right aligned
            frame_info = f"Frame: {self.frame_count} (Skip: {self.frame_skip})"
            text_size = cv2.getTextSize(frame_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(
                original_frame,
                frame_info,
                (w - text_size[0] - 10, 20),  # Right aligned with 10px margin
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            
            # Add prediction history info
            if self.prediction_history:
                history_text = f"History: {self.prediction_history}"
                history_text = history_text[:50]  # Limit text length
                text_size = cv2.getTextSize(history_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(
                    original_frame,
                    history_text,
                    (w - text_size[0] - 10, 40),  # Right aligned with 10px margin
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
            
            # Add cooldown info
            current_time = time.time()
            cooldown_remaining = max(0, self.prediction_cooldown - (current_time - self.last_prediction_time))
            cooldown_text = f"Cooldown: {cooldown_remaining:.2f}s"
            text_size = cv2.getTextSize(cooldown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(
                original_frame,
                cooldown_text,
                (w - text_size[0] - 10, 60),  # Right aligned with 10px margin
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
        
        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Use the original frame for drawing and display
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the original frame
                self.mp_drawing.draw_landmarks(
                    original_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract hand landmarks
                pts = []
                for lm in hand_landmarks.landmark:
                    h, w, c = original_frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pts.append([cx, cy])
                
                # Get the hand region
                x_min = min(pt[0] for pt in pts)
                y_min = min(pt[1] for pt in pts)
                x_max = max(pt[0] for pt in pts)
                y_max = max(pt[1] for pt in pts)
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_max = min(frame.shape[0], y_max + padding)
                
                # Extract the hand image
                hand_img = frame[y_min:y_max, x_min:x_max].copy()
                
                # Draw the hand bounding box if in debug mode
                if self.debug_mode:
                    cv2.rectangle(original_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Number the landmarks if in debug mode
                    for i, pt in enumerate(pts):
                        cv2.putText(
                            original_frame,
                            str(i),
                            (pt[0], pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA
                        )
                
                # Only predict if we have a valid hand image
                if hand_img.size > 0 and hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                    # Check if enough time has passed since the last prediction
                    current_time = time.time()
                    if current_time - self.last_prediction_time >= self.prediction_cooldown:
                        # Make a prediction
                        predicted_symbol = self.predict(hand_img, pts)
                        
                        # Show prediction confidence in debug mode
                        if self.debug_mode and hasattr(self, 'last_confidence'):
                            confidence_text = f"Confidence: {self.last_confidence:.2f}"
                            text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            cv2.putText(
                                original_frame,
                                confidence_text,
                                (w - text_size[0] - 10, 80),  # Right aligned with 10px margin
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA
                            )
                        
                        if predicted_symbol:
                            # Add to prediction history
                            self.prediction_history.append(predicted_symbol)
                            if len(self.prediction_history) > self.history_size:
                                self.prediction_history.pop(0)
                            
                            # Get the most common prediction from history
                            if len(self.prediction_history) >= 3:
                                from collections import Counter
                                counter = Counter(self.prediction_history)
                                most_common = counter.most_common(1)[0][0]
                                
                                # Update current symbol if it's consistent
                                if most_common == self.current_symbol:
                                    self.count_same_symbol += 1
                                else:
                                    self.current_symbol = most_common
                                    self.count_same_symbol = 1
                                
                                # If the same symbol is detected multiple times, emit it
                                if self.count_same_symbol >= 2 and self.current_symbol != self.prev_symbol:
                                    # Emit the prediction signal
                                    self.prediction_signal.emit(self.current_symbol)
                                    
                                    # Handle special characters
                                    if self.current_symbol == "Backspace" and self.current_word:
                                        # Remove the last character from the current word
                                        self.current_word = self.current_word[:-1]
                                        # Update word suggestions
                                        suggestions = self.get_word_suggestions(self.current_word)
                                        self.suggestions_signal.emit(suggestions)
                                        # Update the word signal
                                        self.word_signal.emit(self.output_text + self.current_word)
                                    elif self.current_symbol == " " or self.current_symbol == "next":
                                        # Add the current word to the output text
                                        if self.current_word:
                                            self.output_text += self.current_word + " "
                                            self.current_word = ""
                                            # Reset suggestions
                                            self.suggestions_signal.emit(["", "", "", ""])
                                            # Update the word signal
                                            self.word_signal.emit(self.output_text)
                                    elif len(self.current_symbol) == 1:  # Regular character
                                        # Add the character to the current word
                                        self.current_word += self.current_symbol
                                        # Update word suggestions
                                        suggestions = self.get_word_suggestions(self.current_word)
                                        self.suggestions_signal.emit(suggestions)
                                        # Update the word signal
                                        self.word_signal.emit(self.output_text + self.current_word)
                                    
                                    self.prev_symbol = self.current_symbol
                                    self.count_same_symbol = 0
                            else:
                                # If we don't have enough history, use the current prediction
                                if predicted_symbol == self.current_symbol:
                                    self.count_same_symbol += 1
                                else:
                                    self.current_symbol = predicted_symbol
                                    self.count_same_symbol = 1
                                
                                # If the same symbol is detected multiple times, emit it
                                if self.count_same_symbol >= 3 and self.current_symbol != self.prev_symbol:
                                    # Emit the prediction signal
                                    self.prediction_signal.emit(self.current_symbol)
                                    
                                    # Handle special characters
                                    if self.current_symbol == "Backspace" and self.current_word:
                                        # Remove the last character from the current word
                                        self.current_word = self.current_word[:-1]
                                        # Update word suggestions
                                        suggestions = self.get_word_suggestions(self.current_word)
                                        self.suggestions_signal.emit(suggestions)
                                        # Update the word signal
                                        self.word_signal.emit(self.output_text + self.current_word)
                                    elif self.current_symbol == " " or self.current_symbol == "next":
                                        # Add the current word to the output text
                                        if self.current_word:
                                            self.output_text += self.current_word + " "
                                            self.current_word = ""
                                            # Reset suggestions
                                            self.suggestions_signal.emit(["", "", "", ""])
                                            # Update the word signal
                                            self.word_signal.emit(self.output_text)
                                    elif len(self.current_symbol) == 1:  # Regular character
                                        # Add the character to the current word
                                        self.current_word += self.current_symbol
                                        # Update word suggestions
                                        suggestions = self.get_word_suggestions(self.current_word)
                                        self.suggestions_signal.emit(suggestions)
                                        # Update the word signal
                                        self.word_signal.emit(self.output_text + self.current_word)
                                    
                                    self.prev_symbol = self.current_symbol
                                    self.count_same_symbol = 0
                            
                            self.last_prediction_time = current_time
                    
                    # Draw the current symbol on the frame
                    if self.current_symbol:
                        cv2.putText(
                            original_frame,
                            f"Detected: {self.current_symbol}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
                        
                        # Also draw the current word being built
                        if self.current_word:
                            cv2.putText(
                                original_frame,
                                f"Word: {self.current_word}",
                                (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 0),
                                2,
                                cv2.LINE_AA
                            )
        
        return original_frame
    
    def predict(self, test_image, pts):
        """Predict the sign language character from a hand image"""
        # Resize the image to match model input size (400x400)
        test_image = cv2.resize(test_image, (400, 400))
        
        # Convert to float32 and normalize
        test_image = test_image.astype('float32') / 255.0
        
        # Reshape for model input
        black = test_image.reshape(1, 400, 400, 3)
        
        try:
            # Make prediction with reduced verbosity
            with tf.device('/CPU:0'):  # Force CPU to avoid GPU-related warnings
                prob = np.array(self.model.predict(black, verbose=0)[0], dtype='float32')
            
            # Store the highest confidence value for debug display
            if hasattr(self, 'debug_mode') and self.debug_mode:
                self.last_confidence = np.max(prob)
            
            ch1 = np.argmax(prob, axis=0)
            prob[ch1] = 0
            ch2 = np.argmax(prob, axis=0)
            prob[ch2] = 0
            ch3 = np.argmax(prob, axis=0)
            prob[ch3] = 0
            
            # Rest of the prediction logic
            pl = [ch1, ch2]
            
            # Condition for [Aemnst]
            l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
            if pl in l:
                if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                    ch1 = 0

            # condition for [o][s]
            l = [[2, 2], [2, 1]]
            if pl in l:
                if (pts[5][0] < pts[4][0]):
                    ch1 = 0

            # condition for [c0][aemnst]
            l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
                    ch1 = 2

            # condition for [c0][aemnst]
            l = [[6, 0], [6, 6], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(pts[8], pts[16]) < 52:
                    ch1 = 2

            # condition for [gh][bdfikruvw]
            l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
                    ch1 = 3

            # con for [gh][l]
            l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[4][0] > pts[0][0]:
                    ch1 = 3

            # con for [gh][pqz]
            l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[2][1] + 15 < pts[16][1]:
                    ch1 = 3

            # con for [l][x]
            l = [[6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(pts[4], pts[11]) > 55:
                    ch1 = 4

            # con for [l][d]
            l = [[1, 4], [1, 6], [1, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(pts[4], pts[11]) > 50) and (
                        pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                        pts[20][1]):
                    ch1 = 4

            # con for [l][gh]
            l = [[3, 6], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (pts[4][0] < pts[0][0]):
                    ch1 = 4

            # con for [l][c0]
            l = [[2, 2], [2, 5], [2, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (pts[1][0] < pts[12][0]):
                    ch1 = 4

            # con for [gh][z]
            l = [[3, 6], [3, 5], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
                    ch1 = 5

            # con for [gh][pq]
            l = [[3, 2], [3, 1], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
                    ch1 = 5

            # con for [l][pqz]
            l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[4][0] > pts[0][0]:
                    ch1 = 5

            # con for [pqz][aemnst]
            l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
                    ch1 = 5

            # con for [pqz][yj]
            l = [[5, 7], [5, 2], [5, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[3][0] < pts[0][0]:
                    ch1 = 7

            # con for [l][yj]
            l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[6][1] < pts[8][1]:
                    ch1 = 7

            # con for [x][yj]
            l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[18][1] > pts[20][1]:
                    ch1 = 7

            # condition for [x][aemnst]
            l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[5][0] > pts[16][0]:
                    ch1 = 6

            # condition for [yj][x]
            l = [[7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
                    ch1 = 6

            # condition for [c0][x]
            l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(pts[8], pts[16]) > 50:
                    ch1 = 6

            # con for [l][x]
            l = [[4, 6], [4, 2], [4, 1], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(pts[4], pts[11]) < 60:
                    ch1 = 6

            # con for [x][d]
            l = [[1, 4], [1, 6], [1, 0], [1, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[5][0] - pts[4][0] - 15 > 0:
                    ch1 = 6

            # con for [b][pqz]
            l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
                 [6, 3], [6, 4], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                    ch1 = 1

            # con for [f][pqz]
            l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
                 [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                        pts[18][1] > pts[20][1]):
                    ch1 = 1

            l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                        pts[18][1] > pts[20][1]):
                    ch1 = 1

            # con for [d][pqz]
            l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                     pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
                    ch1 = 1

            l = [[4, 1], [4, 2], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(pts[4], pts[11]) < 50) and (
                        pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                        pts[20][1]):
                    ch1 = 1

            l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                     pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
                    ch1 = 1

            l = [[6, 6], [6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[5][0] - pts[4][0] - 15 < 0:
                    ch1 = 1

            # con for [i][pqz]
            l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                     pts[18][1] > pts[20][1])):
                    ch1 = 1

            # con for [yj][bfdi]
            l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if (pts[4][0] < pts[5][0] + 15) and (
                (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                 pts[18][1] > pts[20][1])):
                    ch1 = 7

            # con for [uvr]
            l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and
                     pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
                    ch1 = 1

            # con for [w]
            fg = 13
            l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and
                        pts[0][0] + fg < pts[20][0]) and not (
                        pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and self.distance(pts[4], pts[11]) < 50:
                    ch1 = 1

            # con for [w]
            l = [[5, 0], [5, 5], [0, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
                    ch1 = 1

            # -------------------------condn for 8 groups  ends

            # -------------------------condn for subgroups  starts
            #
            if ch1 == 0:
                ch1 = 'S'
                if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                    ch1 = 'A'
                if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
                    ch1 = 'T'
                if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                    ch1 = 'E'
                if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1]:
                    ch1 = 'M'
                if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                    ch1 = 'N'

            if ch1 == 2:
                if self.distance(pts[12], pts[4]) > 42:
                    ch1 = 'C'
                else:
                    ch1 = 'O'

            if ch1 == 3:
                if (self.distance(pts[8], pts[12])) > 72:
                    ch1 = 'G'
                else:
                    ch1 = 'H'

            if ch1 == 7:
                if self.distance(pts[8], pts[4]) > 42:
                    ch1 = 'Y'
                else:
                    ch1 = 'J'

            if ch1 == 4:
                ch1 = 'L'

            if ch1 == 6:
                ch1 = 'X'

            if ch1 == 5:
                if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                    if pts[8][1] < pts[5][1]:
                        ch1 = 'Z'
                    else:
                        ch1 = 'Q'
                else:
                    ch1 = 'P'

            if ch1 == 1:
                if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                    ch1 = 'B'
                if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                    ch1 = 'D'
                if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                    ch1 = 'F'
                if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                    ch1 = 'I'
                if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
                    ch1 = 'W'
                if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
                    ch1 = 'K'
                if ((self.distance(pts[8], pts[12]) - self.distance(pts[6], pts[10])) < 8) and (
                        pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                        pts[20][1]):
                    ch1 = 'U'
                if ((self.distance(pts[8], pts[12]) - self.distance(pts[6], pts[10])) >= 8) and (
                        pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                        pts[20][1]) and (pts[4][1] > pts[9][1]):
                    ch1 = 'V'

                if (pts[8][0] > pts[12][0]) and (
                        pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                        pts[20][1]):
                    ch1 = 'R'

            if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
                if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                    ch1=" "

            if ch1 == 'E' or ch1=='Y' or ch1=='B':
                if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                    ch1="next"

            if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
                if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and (pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]) and (pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
                    ch1 = 'Backspace'
            
            # Convert to string if it's a number
            if isinstance(ch1, (int, np.integer)):
                ch1 = str(ch1)
            
            # Return the predicted character
            return ch1
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def distance(self, x, y):
        """Calculate Euclidean distance between two points"""
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))
    
    def get_word_suggestions(self, current_input):
        """
        Get word suggestions based on the current input.
        Returns a list of up to 4 suggestions.
        """
        if not current_input or current_input.strip() == "":
            return ["", "", "", ""]
        
        try:
            # Use enchant dictionary for better suggestions
            import enchant
            d = enchant.Dict("en-US")
            
            # Get suggestions from enchant
            suggestions = []
            
            # First check if the current input is a valid word
            if d.check(current_input.lower()):
                # If it's a valid word, include it as the first suggestion
                suggestions.append(current_input.lower())
                
                # Get related words
                related = d.suggest(current_input.lower())
                # Add related words that are not the same as the current input
                for word in related:
                    if word.lower() != current_input.lower() and word.lower() not in suggestions:
                        suggestions.append(word.lower())
            else:
                # If it's not a valid word, get suggestions
                related = d.suggest(current_input.lower())
                for word in related:
                    if word.lower().startswith(current_input.lower()) and word.lower() not in suggestions:
                        suggestions.append(word.lower())
                
                # If we don't have enough prefix matches, add other suggestions
                if len(suggestions) < 4:
                    for word in related:
                        if word.lower() not in suggestions:
                            suggestions.append(word.lower())
            
            # Return up to 4 suggestions, pad with empty strings if needed
            result = suggestions[:4]
            while len(result) < 4:
                result.append("")
            
            # Convert to uppercase for consistency
            return [s.upper() for s in result]
        except Exception as e:
            print(f"Word suggestion error: {str(e)}")
            # Fallback to NLTK if enchant is not available or has an error
            try:
                # Get English words from NLTK
                english_words = set(word.lower() for word in nltk_words.words())
                
                # Find words that start with the current input
                matching_words = [word for word in english_words if word.startswith(current_input.lower())]
                
                # Sort by length (prefer shorter words first)
                matching_words.sort(key=len)
                
                # Return up to 4 suggestions, pad with empty strings if needed
                suggestions = matching_words[:4]
                while len(suggestions) < 4:
                    suggestions.append("")
                
                return [s.upper() for s in suggestions]
            except Exception as e2:
                print(f"NLTK fallback error: {str(e2)}")
                return ["", "", "", ""]
    
    def stop(self):
        """Stop the video thread"""
        self.running = False
        self.wait()


class LicenseDialog(QWidget):
    """Dialog for entering license keys"""
    
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Enter License Key")
        self.setFixedSize(400, 150)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel("Please enter your license key to activate the full version:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # License key input
        self.license_input = QTextEdit()
        self.license_input.setFixedHeight(50)
        layout.addWidget(self.license_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.activate_btn = QPushButton("Activate")
        self.cancel_btn = QPushButton("Cancel")
        
        button_layout.addWidget(self.activate_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect buttons
        self.activate_btn.clicked.connect(self.activate_license)
        self.cancel_btn.clicked.connect(self.close)
        
        # License validation result
        self.license_valid = False
        self.license_key = ""
    
    def activate_license(self):
        """Validate and activate the license key"""
        self.license_key = self.license_input.toPlainText().strip()
        if len(self.license_key) == 32:  # Simple validation
            self.license_valid = True
            self.close()
        else:
            QMessageBox.warning(self, "Invalid License", "The license key you entered is invalid. Please check and try again.")

class SignLanguageApp(QMainWindow):
    """Main application window for Sign Language to Text conversion"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize instance variables
        self.video_thread = None
        self.str_output = ""
        self.word = ""
        self.debug_mode = False
        
        # Set application icon
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        
        # Initialize the license manager
        self.license_manager = LicenseManager()
        
        # Check license validity
        self.check_license()
        
        # Set up the UI
        self.init_ui()
        
        # Initialize text-to-speech engine
        self.init_tts_engine()
        
        # Show trial popup
        self.show_trial_popup()
    
    def check_license(self):
        """Check license validity and handle trial period"""
        valid, message = self.license_manager.validate_license()
        
        if not valid:
            # Trial expired - show message and exit
            QMessageBox.critical(self, "License Error", message)
            sys.exit(1)
        
        # If we have a trial message, display it in the title bar
        if message:
            self.setWindowTitle(f"Sign Language Converter - {message}")
        else:
            self.setWindowTitle("Sign Language Converter - Licensed Version")
    
    def init_ui(self):
        """Initialize the user interface"""
        # Set window properties
        self.setWindowTitle("Sign Language to Text and Speech Conversion")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create header with title
        header_layout = QHBoxLayout()
        title_label = QLabel("Sign Language to Text and Speech Conversion")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: #4a86e8;")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        main_layout.addLayout(header_layout)
        
        # Create menu layout for buttons with center alignment
        menu_layout = QHBoxLayout()
        menu_layout.setAlignment(Qt.AlignCenter)  # Center align all buttons
        
        # Start/Stop camera button
        self.camera_button = QPushButton("▶️ Start Camera")
        self.camera_button.setFixedWidth(120)  # Shortened width
        self.camera_button.clicked.connect(self.toggle_camera)
        menu_layout.addWidget(self.camera_button)
        
        # Debug mode button
        self.debug_button = QPushButton("🐞 Debug Off")
        self.debug_button.setFixedWidth(100)  # Fixed width
        self.debug_button.clicked.connect(self.toggle_debug_mode)
        menu_layout.addWidget(self.debug_button)
        
        # Dark/Light mode toggle button
        self.dark_mode_button = QPushButton("🌙 Dark Mode")
        self.dark_mode_button.setFixedWidth(100)  # Fixed width
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)
        menu_layout.addWidget(self.dark_mode_button)
        
        # Instructions button
        self.instructions_button = QPushButton("📋 Instructions")
        self.instructions_button.setFixedWidth(100)  # Shortened width
        self.instructions_button.clicked.connect(self.toggle_instructions)
        
        # About button
        self.about_button = QPushButton("ℹ️ About")
        self.about_button.setFixedWidth(80)  # Shortened width
        self.about_button.clicked.connect(self.toggle_about)
        menu_layout.addWidget(self.about_button)
        
        # Controls button
        self.controls_button = QPushButton("🎮 Controls")
        self.controls_button.setFixedWidth(80)  # Shortened width
        self.controls_button.clicked.connect(self.toggle_controls)
        menu_layout.addWidget(self.controls_button)
        
        main_layout.addLayout(menu_layout)
        
        # Create info panels (initially hidden)
        self.instructions_panel = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(self.instructions_panel)
        
        # Create a scroll area for instructions to handle large content
        instructions_scroll = QScrollArea()
        instructions_scroll.setWidgetResizable(True)
        instructions_scroll.setFrameShape(QFrame.NoFrame)
        
        instructions_content = QWidget()
        instructions_scroll_layout = QVBoxLayout(instructions_content)
        
        instructions_text = QLabel(
            "1. Click 'Start Camera' to begin detecting sign language\n"
            "2. Show hand signs in front of the camera\n"
            "3. The detected symbol will appear under 'Current Symbol'\n"
            "4. Click suggested words to autocomplete\n"
            "5. Use 'Speak' for text-to-speech, 'Clear' to reset"
        )
        instructions_scroll_layout.addWidget(instructions_text)
        
        instructions_scroll.setWidget(instructions_content)
        instructions_layout.addWidget(instructions_scroll)
        
        self.instructions_panel.setVisible(False)
        main_layout.addWidget(self.instructions_panel)
        
        self.about_panel = QGroupBox("About")
        about_layout = QVBoxLayout(self.about_panel)
        about_text = QLabel(
            "This app converts sign language to text and speech in real-time using computer vision and machine learning."
        )
        about_layout.addWidget(about_text)
        self.about_panel.setVisible(False)
        main_layout.addWidget(self.about_panel)
        
        self.controls_panel = QGroupBox("Controls")
        controls_layout = QVBoxLayout(self.controls_panel)
        controls_text = QLabel(
            "- Start/Stop: Begin/End detection\n"
            "- Speak: Text-to-speech output\n"
            "- Clear: Reset output\n"
            "- Word Suggestions: Click to autocomplete"
        )
        controls_layout.addWidget(controls_text)
        
        self.controls_panel.setVisible(False)
        main_layout.addWidget(self.controls_panel)
        
        # Create main content area with camera feed and controls
        content_layout = QHBoxLayout()
        
        # Camera feed on the left (3/4 of width)
        camera_group = QGroupBox("Camera Feed")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #f0f0f0; border-radius: 10px;")
        camera_layout.addWidget(self.camera_label)
        
        # Controls on the right (1/4 of width)
        controls_group = QWidget()
        controls_group_layout = QVBoxLayout(controls_group)
        
        # Current symbol display
        symbol_group = QGroupBox("Current Symbol")
        symbol_layout = QVBoxLayout(symbol_group)
        self.symbol_label = QLabel("")
        self.symbol_label.setAlignment(Qt.AlignCenter)
        self.symbol_label.setFont(QFont("Arial", 36, QFont.Bold))
        self.symbol_label.setStyleSheet(
            "background-color: #4a86e8; color: white; border-radius: 10px; padding: 10px;"
        )
        self.symbol_label.setMinimumHeight(80)
        symbol_layout.addWidget(self.symbol_label)
        controls_group_layout.addWidget(symbol_group)
        
        # Word suggestions
        suggestions_group = QGroupBox("Word Suggestions")
        suggestions_layout = QHBoxLayout(suggestions_group)
        self.suggestion_buttons = []
        for i in range(4):
            button = QPushButton("")
            button.setEnabled(False)  # Initially disabled
            button.clicked.connect(lambda checked, idx=i: self.use_suggestion(idx))
            suggestions_layout.addWidget(button)
            self.suggestion_buttons.append(button)
        controls_group_layout.addWidget(suggestions_group)
        
        # Output text - with reduced height
        output_group = QGroupBox("Output Text")
        output_layout = QVBoxLayout(output_group)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(80)  # Reduced height
        self.output_text.setMaximumHeight(100)  # Set maximum height
        self.output_text.setStyleSheet(
            "background-color: #4a86e8; color: white; border-radius: 10px; font-size: 18px;"
        )
        output_layout.addWidget(self.output_text)
        controls_group_layout.addWidget(output_group)
        
        # Clear and Speak buttons
        buttons_layout = QHBoxLayout()
        self.clear_button = QPushButton("🗑️ Clear Text")
        self.clear_button.clicked.connect(self.clear_text)
        buttons_layout.addWidget(self.clear_button)
        
        self.speak_button = QPushButton("🔊 Speak Text")
        self.speak_button.clicked.connect(self.speak_text)
        buttons_layout.addWidget(self.speak_button)
        controls_group_layout.addLayout(buttons_layout)
        
        # Add hand signs reference image below the buttons
        hand_signs_group = QGroupBox("Hand Signs Reference")
        hand_signs_layout = QVBoxLayout(hand_signs_group)
        
        # Load and display the signs.jpg image
        image_path = os.path.join(os.path.dirname(__file__), 'signs.jpg')
        hand_signs_image = QLabel()
        pixmap = QPixmap(image_path)
        
        # Scale the image to fit the width while maintaining aspect ratio
        hand_signs_image.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        hand_signs_image.setAlignment(Qt.AlignCenter)
        hand_signs_layout.addWidget(hand_signs_image)
        
        controls_group_layout.addWidget(hand_signs_group)
        
        # Add a stretch to push everything up
        controls_group_layout.addStretch(1)
        
        # Add camera and controls to the content layout
        content_layout.addWidget(camera_group, 3)
        content_layout.addWidget(controls_group, 1)
        
        main_layout.addLayout(content_layout)
        
        # Show a placeholder image in the camera feed
        self.show_placeholder_image()
        
        self.setCentralWidget(central_widget)
    
    def init_tts_engine(self):
        """Initialize the text-to-speech engine"""
        self.engine = None
        self.tts_engine = None
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume level
        except Exception as e:
            print(f"Error initializing text-to-speech engine: {e}")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.video_thread is None or not self.video_thread.running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera and processing thread"""
        # Create and configure video thread
        self.video_thread = VideoThread()
        
        # Set debug mode in the video thread
        self.video_thread.debug_mode = self.debug_mode
        
        # Connect signals
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.prediction_signal.connect(self.update_symbol)
        self.video_thread.word_signal.connect(self.update_word)
        self.video_thread.suggestions_signal.connect(self.update_suggestions)
        
        # Start the thread
        self.video_thread.start()
        
        # Update UI
        self.camera_button.setText("⏹️ Stop Camera")
    
    def stop_camera(self):
        """Stop the camera and processing thread"""
        if self.video_thread is not None and self.video_thread.running:
            self.video_thread.stop()
            self.video_thread = None
            
            # Update UI
            self.camera_button.setText("▶️ Start Camera")
            self.show_placeholder_image()
    
    def show_placeholder_image(self):
        """Display a placeholder image when camera is off"""
        placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder_image[:] = (245, 245, 245)  # Light gray background
        
        # Add text to the placeholder image
        cv2.putText(placeholder_image, "Camera Off", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Convert to QImage and display
        self.update_image(placeholder_image)
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Update the image in the camera feed label"""
        # Convert the frame to RGB for display
        rgb_frame = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale the image to fit the label
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set the pixmap to the label
        self.camera_label.setPixmap(pixmap)
    
    @pyqtSlot(str)
    def update_symbol(self, symbol):
        """Update the current symbol display"""
        self.symbol_label.setText(symbol)
    
    @pyqtSlot(str)
    def update_word(self, word):
        """Update the current word display"""
        self.output_text.setText(word)
    
    @pyqtSlot(list)
    def update_suggestions(self, suggestions):
        """Update the word suggestions"""
        for i, suggestion in enumerate(suggestions):
            self.suggestion_buttons[i].setText(suggestion)
            self.suggestion_buttons[i].setEnabled(bool(suggestion.strip()))
    
    def show_trial_popup(self):
        """Show attractive trial popup with persuasive purchase options"""
        valid, message = self.license_manager.validate_license()
        
        if not valid:
            QMessageBox.critical(self, "License Error", message)
            sys.exit(1)
        
        # Create custom dialog for trial information
        dialog = QDialog(self)
        dialog.setWindowTitle("Welcome to Sign Language Converter Pro")
        dialog.setFixedSize(900, 650)  # Increased width and fixed height
        dialog.setWindowIcon(self.windowIcon())
        # Disable resizing
        dialog.setWindowFlags(dialog.windowFlags() | Qt.MSWindowsFixedSizeDialogHint)
        
        # Set dialog stylesheet for modern look
        dialog.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 10px;
            }
            QLabel {
                color: #333;
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Header with logo and title
        header_layout = QHBoxLayout()
        
        # Use app icon as logo
        logo_label = QLabel()
        if os.path.exists(ICON_PATH):
            logo_pixmap = QPixmap(ICON_PATH).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
        header_layout.addWidget(logo_label)
        
        # Add title and license status
        title_layout = QVBoxLayout()
        title_label = QLabel("Sign Language Converter Pro")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #4a86e8;")
        title_layout.addWidget(title_label)
        
        # License status with custom styling
        status_label = QLabel(f"License Status: {message}")
        if "Trial" in message:
            days_left = message.split(":")[1].strip()
            status_label.setText(f"✨ Trial Version - {days_left} ✨")
            status_label.setStyleSheet("font-size: 18px; color: #ff9800; font-weight: bold;")
        else:
            status_label.setText("✅ Licensed Version")
            status_label.setStyleSheet("font-size: 18px; color: #4caf50; font-weight: bold;")
        title_layout.addWidget(status_label)
        header_layout.addLayout(title_layout, 1)
        layout.addLayout(header_layout)
        
        # Add horizontal separator with gradient styling
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4a86e8, stop:1 #9dc3ff); height: 2px;")
        layout.addWidget(line)
        
        # Create tab widget for better organization
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                padding: 12px 30px;  /* Increased padding for wider tabs */
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-size: 15px;     /* Slightly larger text */
                min-width: 180px;    /* Minimum width for tabs */
            }
            QTabBar::tab:selected {
                background-color: #4a86e8;
                color: white;
                font-weight: bold;
            }
        """)
        
        # Get started tab
        get_started_tab = QWidget()
        get_started_layout = QVBoxLayout(get_started_tab)
        
        quick_start_group = QGroupBox("Quick Start Guide")
        quick_start_layout = QVBoxLayout(quick_start_group)
        
        instructions = QLabel("""
        <ol>
            <li><b>Start the camera</b> by clicking the "Start Camera" button</li>
            <li><b>Position your hand</b> in front of the camera with good lighting</li>
            <li><b>Make sign gestures</b> and hold them steady for about 1 second</li>
            <li><b>Watch as your signs</b> are converted to text in real-time</li>
            <li><b>Use the word suggestions</b> to speed up your communication</li>
            <li><b>Click "Speak"</b> to have your text read aloud</li>
        </ol>
        
        <p><b>💡 Pro Tip:</b> Use a plain, contrasting background behind your hand for best recognition results.</p>
        """)
        instructions.setTextFormat(Qt.RichText)
        instructions.setWordWrap(True)
        quick_start_layout.addWidget(instructions)
        get_started_layout.addWidget(quick_start_group)
        
        # Debug features description
        debug_group = QGroupBox("Debug Mode Features")
        debug_layout = QVBoxLayout(debug_group)
        debug_info = QLabel("""
        <p>Click the <b>"Debug On"</b> button to access advanced features:</p>
        <ul>
            <li>Hand landmark visualization with numbered points</li>
            <li>Real-time confidence scores for gesture recognition</li>
            <li>Frame processing statistics and optimization metrics</li>
            <li>Gesture prediction history for troubleshooting</li>
        </ul>
        """)
        debug_info.setTextFormat(Qt.RichText)
        debug_info.setWordWrap(True)
        debug_layout.addWidget(debug_info)
        get_started_layout.addWidget(debug_group)
        
        tab_widget.addTab(get_started_tab, "📝 Get Started")
        
        # Pro features tab - enhanced with better visuals and persuasive copy
        pro_features_tab = QWidget()
        pro_features_layout = QVBoxLayout(pro_features_tab)
        
        features_label = QLabel("""
        <h3 style="color: #4a86e8; text-align: center; margin-bottom: 20px;">🌟 Unlock Premium Features with a License! 🌟</h3>
        
        <p style="font-weight: bold; color: #ff5722; font-size: 16px;">Why upgrade to the full version?</p>
        
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr style="background-color: #f0f7ff;">
                <td style="padding: 12px; border: 1px solid #ccc; font-weight: bold;">Feature</td>
                <td style="padding: 12px; border: 1px solid #ccc; font-weight: bold;">Trial Version</td>
                <td style="padding: 12px; border: 1px solid #ccc; font-weight: bold;">Full Version</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ccc;">Usage Period</td>
                <td style="padding: 12px; border: 1px solid #ccc;">30 Days</td>
                <td style="padding: 12px; border: 1px solid #ccc; color: #4caf50; font-weight: bold;">✓ Unlimited</td>
            </tr>
            <tr style="background-color: #f0f7ff;">
                <td style="padding: 12px; border: 1px solid #ccc;">Advanced Word Prediction</td>
                <td style="padding: 12px; border: 1px solid #ccc;">Basic</td>
                <td style="padding: 12px; border: 1px solid #ccc; color: #4caf50; font-weight: bold;">✓ Enhanced AI</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ccc;">Gesture Recognition Accuracy</td>
                <td style="padding: 12px; border: 1px solid #ccc;">Standard</td>
                <td style="padding: 12px; border: 1px solid #ccc; color: #4caf50; font-weight: bold;">✓ High Precision</td>
            </tr>
            <tr style="background-color: #f0f7ff;">
                <td style="padding: 12px; border: 1px solid #ccc;">Export & Save Translations</td>
                <td style="padding: 12px; border: 1px solid #ccc;">✗ No</td>
                <td style="padding: 12px; border: 1px solid #ccc; color: #4caf50; font-weight: bold;">✓ Yes</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ccc;">Low Light Performance</td>
                <td style="padding: 12px; border: 1px solid #ccc;">Limited</td>
                <td style="padding: 12px; border: 1px solid #ccc; color: #4caf50; font-weight: bold;">✓ Enhanced</td>
            </tr>
            <tr style="background-color: #f0f7ff;">
                <td style="padding: 12px; border: 1px solid #ccc;">Custom Gesture Training</td>
                <td style="padding: 12px; border: 1px solid #ccc;">✗ No</td>
                <td style="padding: 12px; border: 1px solid #ccc; color: #4caf50; font-weight: bold;">✓ Yes</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ccc;">Premium Support</td>
                <td style="padding: 12px; border: 1px solid #ccc;">✗ No</td>
                <td style="padding: 12px; border: 1px solid #ccc; color: #4caf50; font-weight: bold;">✓ Yes</td>
            </tr>
        </table>
        
        <div style="background-color: #fff4e5; border-left: 4px solid #ff9800; padding: 15px; margin: 15px 0; border-radius: 4px;">
            <p style="font-weight: bold; color: #ff5722; margin: 0 0 10px 0;">🔥 LIMITED TIME OFFER 🔥</p>
            <p style="margin: 0;">Purchase now and get <span style="font-weight: bold;">25% OFF</span> with code <span style="background-color: #ffeb3b; padding: 3px 6px; border-radius: 3px; font-family: monospace; font-weight: bold;">EARLY25</span></p>
        </div>
        
        <p style="text-align: center; font-style: italic; color: #666; margin-top: 15px;">
            Our full version is perfect for educational institutions, healthcare providers, and individuals who regularly communicate with sign language users.
        </p>
        """)
        features_label.setTextFormat(Qt.RichText)
        features_label.setWordWrap(True)
        
        # Create a scroll area for the features content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.addWidget(features_label)
        scroll_area.setWidget(scroll_content)
        
        pro_features_layout.addWidget(scroll_area)
        
        # Enhanced testimonials section
        testimonials_group = QGroupBox("What Our Users Say")
        testimonials_layout = QVBoxLayout(testimonials_group)
        
        testimonials = QLabel("""
        <blockquote style="border-left: 4px solid #4a86e8; padding-left: 15px; margin: 15px 0; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
            "As an ASL instructor, this software has revolutionized how I teach remote classes. The accuracy is impressive!"
            <footer style="margin-top: 5px; color: #666;">— Sarah J., <cite>Language Professor</cite></footer>
        </blockquote>
        
        <blockquote style="border-left: 4px solid #4a86e8; padding-left: 15px; margin: 15px 0; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
            "The word prediction feature saves me so much time. It's like the software knows what I'm trying to sign!"
            <footer style="margin-top: 5px; color: #666;">— Michael T., <cite>Daily User</cite></footer>
        </blockquote>
        
        <blockquote style="border-left: 4px solid #4a86e8; padding-left: 15px; margin: 15px 0; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
            "After upgrading to the full version, the recognition accuracy improved dramatically. Worth every penny!"
            <footer style="margin-top: 5px; color: #666;">— Lisa R., <cite>Special Education Teacher</cite></footer>
        </blockquote>
        """)
        testimonials.setTextFormat(Qt.RichText)
        testimonials.setWordWrap(True)
        testimonials_layout.addWidget(testimonials)
        
        pro_features_layout.addWidget(testimonials_group)
        tab_widget.addTab(pro_features_tab, "⭐ Pro Features")
        
        # Pricing tab - Simplified with rupees instead of dollars
        pricing_tab = QWidget()
        pricing_layout = QVBoxLayout(pricing_tab)
        
        pricing_info = QLabel("""
        <h3 style="color: #4a86e8; text-align: center; margin-bottom: 20px;">Simple Pricing Options</h3>
        
        <div style="display: flex; justify-content: space-around; margin-bottom: 30px;">
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px; width: 30%; background-color: #f8f9fa;">
                <h4 style="color: #4a86e8; text-align: center; margin-top: 0;">Basic</h4>
                <p style="font-size: 24px; text-align: center; font-weight: bold;">₹999</p>
                <ul style="padding-left: 20px;">
                    <li>Single user license</li>
                    <li>All premium features</li>
                    <li>Email support</li>
        </ul>
            </div>
            
            <div style="border: 2px solid #4a86e8; border-radius: 8px; padding: 20px; width: 30%; background-color: #f0f7ff; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h4 style="color: #4a86e8; text-align: center; margin-top: 0;">Professional</h4>
                <p style="font-size: 24px; text-align: center; font-weight: bold;">₹1,899</p>
                <ul style="padding-left: 20px;">
                    <li>Up to 3 users</li>
                    <li>All premium features</li>
                    <li>Priority support</li>
                    <li>Custom gesture training</li>
                </ul>
            </div>
            
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px; width: 30%; background-color: #f8f9fa;">
                <h4 style="color: #4a86e8; text-align: center; margin-top: 0;">Enterprise</h4>
                <p style="font-size: 24px; text-align: center; font-weight: bold;">Contact Us</p>
                <ul style="padding-left: 20px;">
                    <li>Unlimited users</li>
                    <li>Premium features</li>
                    <li>Priority support</li>
                    <li>Custom integration</li>
                </ul>
            </div>
        </div>
        
        <div style="background-color: #e3f2fd; border-radius: 8px; padding: 15px; margin: 15px 0; text-align: center;">
            <p style="font-weight: bold; margin: 0;">Special offer for educational institutions</p>
        </div>
        """)
        pricing_info.setTextFormat(Qt.RichText)
        pricing_info.setWordWrap(True)
        
        scroll_area_pricing = QScrollArea()
        scroll_area_pricing.setWidgetResizable(True)
        scroll_area_pricing.setFrameShape(QFrame.NoFrame)
        scroll_content_pricing = QWidget()
        scroll_layout_pricing = QVBoxLayout(scroll_content_pricing)
        scroll_layout_pricing.addWidget(pricing_info)
        scroll_area_pricing.setWidget(scroll_content_pricing)
        
        pricing_layout.addWidget(scroll_area_pricing)
        tab_widget.addTab(pricing_tab, "💰 Pricing")
        
        layout.addWidget(tab_widget)
        
        # Buttons area
        buttons_layout = QHBoxLayout()
        
        # Purchase button with special styling
        if message and "Trial" in message:
            purchase_btn = QPushButton("🛒 Upgrade Now")
            purchase_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff5722;
                    color: white;
                    padding: 12px 24px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 16px;
                    min-height: 50px;
                }
                QPushButton:hover {
                    background-color: #f4511e;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    transform: translateY(-2px);
                }
            """)
            purchase_btn.clicked.connect(self.show_purchase_info)
            buttons_layout.addWidget(purchase_btn)
        
        # Continue button
        continue_btn = QPushButton("Continue to Application")
        if not (message and "Trial" in message):
            continue_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px 24px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 16px;
                    min-height: 50px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }
            """)
        continue_btn.clicked.connect(dialog.accept)
        buttons_layout.addWidget(continue_btn)
        
        layout.addLayout(buttons_layout)
        
        # Add countdown timer for trial version with more visual impact
        if message and "Trial" in message:
            try:
                days_left = int(message.split(":")[1].strip().split(" ")[0])
                timer_container = QWidget()
                timer_layout = QVBoxLayout(timer_container)
                
                # Style the container
                timer_container.setStyleSheet("""
                    background-color: #fff8e1;
                    border-left: 4px solid #ff9800;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding: 10px;
                """)
                
                timer_label = QLabel(f"⏳ Trial expires in {days_left} days! ⏳")
                timer_label.setStyleSheet("color: #ff5722; font-weight: bold; font-style: italic; text-align: center; font-size: 14px;")
                timer_label.setAlignment(Qt.AlignCenter)
                timer_layout.addWidget(timer_label)
                
                layout.addWidget(timer_container)
            except:
                pass
        
        dialog.setLayout(layout)
        dialog.exec_()
        
        # Update window title
        if message:
            self.setWindowTitle(f"Sign Language Converter Pro - {message}")
        else:
            self.setWindowTitle("Sign Language Converter Pro - Licensed Version")
    
    def show_purchase_info(self):
        """Show license purchase information"""
        purchase_dialog = QDialog(self)
        purchase_dialog.setWindowTitle("Purchase License")
        purchase_dialog.setFixedSize(600, 500)
        purchase_dialog.setWindowIcon(self.windowIcon())
        # Disable resizing
        purchase_dialog.setWindowFlags(purchase_dialog.windowFlags() | Qt.MSWindowsFixedSizeDialogHint)
        
        # Set dialog stylesheet for modern look
        purchase_dialog.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 10px;
            }
            QLabel {
                color: #333;
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Header with logo
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        if os.path.exists(ICON_PATH):
            logo_pixmap = QPixmap(ICON_PATH).scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
        header_layout.addWidget(logo_label)
        
        # Title
        title_label = QLabel("Purchase Your License Today!")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4a86e8;")
        header_layout.addWidget(title_label, 1)
        layout.addLayout(header_layout)
        
        # Add horizontal separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4a86e8, stop:1 #9dc3ff); height: 2px;")
        layout.addWidget(line)
        
        # Main content
        content_label = QLabel()
        content_label.setWordWrap(True)
        content_label.setTextFormat(Qt.RichText)
        content_label.setText("""
        <h3 style="color: #4a86e8; text-align: center;">Unlock the Full Potential of Sign Language Converter Pro!</h3>
        
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 15px 0;">
            <p style="font-weight: bold; margin-bottom: 10px;">Choose Your License Type:</p>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #bbdefb;">
                    <td style="padding: 10px; border: 1px solid #90caf9; font-weight: bold;">License Type</td>
                    <td style="padding: 10px; border: 1px solid #90caf9; font-weight: bold;">Price</td>
                    <td style="padding: 10px; border: 1px solid #90caf9; font-weight: bold;">Best For</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #90caf9;"><b>Basic</b></td>
                    <td style="padding: 10px; border: 1px solid #90caf9;">₹999</td>
                    <td style="padding: 10px; border: 1px solid #90caf9;">Personal use</td>
                </tr>
                <tr style="background-color: #e3f2fd;">
                    <td style="padding: 10px; border: 1px solid #90caf9;"><b>Professional</b></td>
                    <td style="padding: 10px; border: 1px solid #90caf9;">₹1,899</td>
                    <td style="padding: 10px; border: 1px solid #90caf9;">Small teams (up to 3 users)</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #90caf9;"><b>Enterprise</b></td>
                    <td style="padding: 10px; border: 1px solid #90caf9;">Contact Us</td>
                    <td style="padding: 10px; border: 1px solid #90caf9;">Organizations</td>
                </tr>
            </table>
        </div>
        
        <p style="font-weight: bold;">How to Purchase:</p>
        <ol>
            <li>Visit our website: <a href="http://www.signlanguageconverter.com/license" style="color: #4a86e8;">www.signlanguageconverter.com/license</a></li>
            <li>Select your preferred license type</li>
            <li>Receive your activation key via email</li>
            <li>Enter the key in the application to activate</li>
        </ol>
        
        <p style="margin-top: 15px;">For enterprise licensing:</p>
        <p style="margin: 5px 0;">
            <span style="color: #4a86e8;">✉️</span> Email: <a href="mailto:sales@signlanguageconverter.com" style="color: #4a86e8;">sales@signlanguageconverter.com</a><br>
            <span style="color: #4a86e8;">📞</span> Phone: +1-555-0123-4567
        </p>
        """)
        layout.addWidget(content_label)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Visit website button
        website_btn = QPushButton("🌐 Visit Website")
        website_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
        """)
        website_btn.clicked.connect(lambda: QMessageBox.information(purchase_dialog, "Website", "This would open your web browser to our website."))
        buttons_layout.addWidget(website_btn)
        
        # Enter key button
        key_btn = QPushButton("🔑 Enter License Key")
        key_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        key_btn.clicked.connect(lambda: QMessageBox.information(purchase_dialog, "License Key", "This would open a dialog to enter your license key."))
        buttons_layout.addWidget(key_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(purchase_dialog.accept)
        buttons_layout.addWidget(close_btn)
        
        layout.addLayout(buttons_layout)
        
        purchase_dialog.setLayout(layout)
        purchase_dialog.exec_()
    
    def clear_text(self):
        """Clear the output text"""
        self.output_text.clear()
        if hasattr(self, 'video_thread') and self.video_thread is not None:
            # Reset the output text in the video thread as well
            self.video_thread.output_text = ""
            self.video_thread.current_word = ""
    
    def speak_text(self):
        """Read the output text using text-to-speech"""
        if self.engine is not None:
            text = self.output_text.toPlainText().strip()
            if text:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"Error speaking text: {e}")
                    QMessageBox.warning(self, "Text-to-Speech Error", 
                                     "Could not speak the text. Please check your audio settings.")
            else:
                QMessageBox.information(self, "Text-to-Speech", "No text to speak.")
        else:
            QMessageBox.warning(self, "Text-to-Speech Error", 
                             "Text-to-speech engine not available. Please check your system configuration.")
    
    def toggle_instructions(self):
        """Toggle the instructions panel visibility"""
        visible = self.instructions_panel.isVisible()
        self.instructions_panel.setVisible(not visible)
        self.about_panel.setVisible(False)
        self.controls_panel.setVisible(False)
        
        # Update button style
        if not visible:
            self.instructions_button.setText("📋 Instructions ✓")
            self.about_button.setText("ℹ️ About")
            self.controls_button.setText("🎮 Controls")
        else:
            self.instructions_button.setText("📋 Instructions")
    
    def toggle_about(self):
        """Toggle the about panel visibility"""
        visible = self.about_panel.isVisible()
        self.about_panel.setVisible(not visible)
        self.instructions_panel.setVisible(False)
        self.controls_panel.setVisible(False)
        
        # Update button style
        if not visible:
            self.about_button.setText("ℹ️ About ✓")
            self.instructions_button.setText("📋 Instructions")
            self.controls_button.setText("🎮 Controls")
        else:
            self.about_button.setText("ℹ️ About")
    
    def toggle_controls(self):
        """Toggle the controls panel visibility"""
        visible = self.controls_panel.isVisible()
        self.controls_panel.setVisible(not visible)
        self.instructions_panel.setVisible(False)
        self.about_panel.setVisible(False)
        
        # Update button style
        if not visible:
            self.controls_button.setText("🎮 Controls ✓")
            self.instructions_button.setText("📋 Instructions")
            self.about_button.setText("ℹ️ About")
        else:
            self.controls_button.setText("🎮 Controls")
    
    def toggle_debug_mode(self, state=None):
        """Toggle debug mode"""
        # Update debug mode state
        self.debug_mode = not self.debug_mode
        
        # Update button text based on debug mode state
        if self.debug_mode:
            self.debug_button.setText("🐞 Debug On")
            # Apply debug styling if needed
            self.debug_button.setStyleSheet("background-color: #FFC107; color: black;")
        else:
            self.debug_button.setText("🐞 Debug Off")
            self.debug_button.setStyleSheet("")
            
        # If video thread exists, update its debug mode
        if hasattr(self, 'video_thread') and self.video_thread is not None:
            self.video_thread.debug_mode = self.debug_mode
            
        print(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
    
    def toggle_dark_mode(self):
        """Toggle dark mode"""
        if self.dark_mode_button.text() == "🌙 Dark Mode":
            self.dark_mode_button.setText("☀️ Light Mode")
            self.setStyleSheet("background-color: #2f2f2f; color: white;")
            self.camera_label.setStyleSheet("background-color: #2f2f2f; border-radius: 10px;")
            self.symbol_label.setStyleSheet(
                "background-color: #4a86e8; color: white; border-radius: 10px; padding: 10px;"
            )
            self.output_text.setStyleSheet(
                "background-color: #4a86e8; color: white; border-radius: 10px; font-size: 18px;"
            )
        else:
            self.dark_mode_button.setText("🌙 Dark Mode")
            self.setStyleSheet("background-color: #f0f0f0; color: black;")
            self.camera_label.setStyleSheet("background-color: #f0f0f0; border-radius: 10px;")
            self.symbol_label.setStyleSheet(
                "background-color: #4a86e8; color: white; border-radius: 10px; padding: 10px;"
            )
            self.output_text.setStyleSheet(
                "background-color: #4a86e8; color: white; border-radius: 10px; font-size: 18px;"
            )
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop the video thread if it's running
        if self.video_thread is not None and self.video_thread.running:
            self.video_thread.stop()
        
        # Clean up text-to-speech engine
        if self.engine is not None:
            self.engine.stop()
            
        # Clean up the TTS engine used for speaking
        if hasattr(self, 'tts_engine') and self.tts_engine is not None:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Set app icon globally
    if os.path.exists(ICON_PATH):
        app_icon = QIcon(ICON_PATH)
        app.setWindowIcon(app_icon)
    
    # Create and show the main window
    window = SignLanguageApp()
    window.show()
    
    sys.exit(app.exec_())
