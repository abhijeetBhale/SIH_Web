from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import math

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@dataclass
class AthleteProfile:
    """Athlete profile data structure"""
    athlete_id: str
    name: str
    age: int
    sport: str
    region: str
    phone: str
    created_at: datetime

@dataclass
class PerformanceMetrics:
    """Performance metrics from AI analysis"""
    overall_score: float
    technique_score: float
    posture_score: float
    stability_score: float
    speed_score: float
    form_consistency: float
    injury_risk: str
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]

class MediaPipePoseAnalyzer:
    """Core AI analyzer using MediaPipe for pose estimation"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key body landmarks for different sports
        self.sport_keypoints = {
            'cricket': {
                'bowling': [11, 12, 13, 14, 15, 16, 23, 24],
                'batting': [11, 12, 13, 14, 15, 16, 23, 24, 25, 26],
                'fielding': [11, 12, 13, 14, 23, 24, 25, 26, 27, 28]
            },
            'football': {
                'dribbling': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                'shooting': [11, 12, 23, 24, 25, 26, 27, 28],
                'goalkeeping': [11, 12, 13, 14, 15, 16, 23, 24]
            },
            'athletics': {
                'running': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                'jumping': [11, 12, 23, 24, 25, 26, 27, 28],
                'throwing': [11, 12, 13, 14, 15, 16]
            }
        }

    def analyze_video(self, video_path: str, sport: str, activity: str) -> PerformanceMetrics:
        """Main function to analyze athlete performance video"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        pose_data = []
        stability_scores = []
        technique_scores = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Skip frames for performance (analyze every 5th frame)
            if frame_count % 5 != 0:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = self._extract_landmarks(results.pose_landmarks)
                pose_data.append(landmarks)
                
                stability_score = self._calculate_stability(landmarks)
                technique_score = self._analyze_technique(landmarks, sport, activity)
                
                stability_scores.append(stability_score)
                technique_scores.append(technique_score)
        
        cap.release()
        
        if not pose_data:
            raise ValueError("No pose data detected in video")
        
        return self._generate_performance_metrics(
            pose_data, stability_scores, technique_scores, sport, activity
        )

    def _extract_landmarks(self, pose_landmarks) -> Dict[str, Tuple[float, float, float]]:
        """Extract landmark coordinates"""
        landmarks = {}
        for idx, landmark in enumerate(pose_landmarks.landmark):
            landmarks[idx] = (landmark.x, landmark.y, landmark.z)
        return landmarks

    def _calculate_stability(self, landmarks: Dict) -> float:
        """Calculate body stability score based on pose consistency"""
        if not landmarks:
            return 0.0
        
        hip_left = np.array(landmarks.get(23, [0, 0, 0]))
        hip_right = np.array(landmarks.get(24, [0, 0, 0]))
        
        shoulder_left = np.array(landmarks.get(11, [0, 0, 0]))
        shoulder_right = np.array(landmarks.get(12, [0, 0, 0]))
        shoulder_alignment = 1 - abs(shoulder_left[1] - shoulder_right[1])
        
        stability = (shoulder_alignment * 0.6 + 0.4) * 100
        return min(100, max(0, stability))

    def _analyze_technique(self, landmarks: Dict, sport: str, activity: str) -> float:
        """Analyze sport-specific technique"""
        if sport not in self.sport_keypoints:
            return 50.0
        
        if activity not in self.sport_keypoints[sport]:
            return 50.0
        
        if sport == 'cricket' and activity == 'bowling':
            return self._analyze_cricket_bowling(landmarks)
        elif sport == 'football' and activity == 'shooting':
            return self._analyze_football_shooting(landmarks)
        elif sport == 'athletics' and activity == 'running':
            return self._analyze_running_form(landmarks)
        else:
            return self._generic_posture_analysis(landmarks, self.sport_keypoints[sport][activity])

    def _analyze_cricket_bowling(self, landmarks: Dict) -> float:
        """Cricket bowling specific analysis"""
        scores = []
        
        shoulder = np.array(landmarks.get(12, [0, 0, 0]))
        elbow = np.array(landmarks.get(14, [0, 0, 0]))
        wrist = np.array(landmarks.get(16, [0, 0, 0]))
        
        if np.any(shoulder) and np.any(elbow) and np.any(wrist):
            arm_vector1 = elbow - shoulder
            arm_vector2 = wrist - elbow
            
            dot_product = np.dot(arm_vector1, arm_vector2)
            norms = np.linalg.norm(arm_vector1) * np.linalg.norm(arm_vector2)
            
            if norms > 0:
                angle = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
                angle_degrees = np.degrees(angle)
                arm_score = max(0, 100 - abs(170 - angle_degrees) * 2)
                scores.append(arm_score)
        
        hip_left = np.array(landmarks.get(23, [0, 0, 0]))
        hip_right = np.array(landmarks.get(24, [0, 0, 0]))
        
        if np.any(hip_left) and np.any(hip_right):
            hip_alignment = 100 - abs(hip_left[1] - hip_right[1]) * 500
            scores.append(max(0, hip_alignment))
        
        return np.mean(scores) if scores else 50.0

    def _analyze_football_shooting(self, landmarks: Dict) -> float:
        """Football shooting technique analysis"""
        scores = []
        
        ankle_right = np.array(landmarks.get(28, [0, 0, 0]))
        knee_right = np.array(landmarks.get(26, [0, 0, 0]))
        
        if np.any(knee_right) and np.any(ankle_right):
            leg_vector = ankle_right - knee_right
            vertical_alignment = abs(leg_vector[0])
            alignment_score = max(0, 100 - vertical_alignment * 300)
            scores.append(alignment_score)
        
        hip_center = (np.array(landmarks.get(23, [0, 0, 0])) + 
                     np.array(landmarks.get(24, [0, 0, 0]))) / 2
        
        if np.any(hip_center):
            balance_score = 75
            scores.append(balance_score)
        
        return np.mean(scores) if scores else 50.0

    def _analyze_running_form(self, landmarks: Dict) -> float:
        """Running form analysis"""
        scores = []
        
        hip_left = np.array(landmarks.get(23, [0, 0, 0]))
        hip_right = np.array(landmarks.get(24, [0, 0, 0]))
        knee_left = np.array(landmarks.get(25, [0, 0, 0]))
        
        if np.any(hip_left) and np.any(hip_right):
            hip_stability = 100 - abs(hip_left[1] - hip_right[1]) * 400
            scores.append(max(0, hip_stability))
        
        if np.any(knee_left) and np.any(hip_left):
            knee_lift = abs(knee_left[1] - hip_left[1])
            lift_score = min(100, knee_lift * 300)
            scores.append(lift_score)
        
        return np.mean(scores) if scores else 50.0

    def _generic_posture_analysis(self, landmarks: Dict, relevant_points: List[int]) -> float:
        """Generic posture analysis"""
        if not relevant_points:
            return 50.0
        
        scores = []
        
        for i in range(0, len(relevant_points) - 1, 2):
            if i + 1 < len(relevant_points):
                point1 = landmarks.get(relevant_points[i])
                point2 = landmarks.get(relevant_points[i + 1])
                
                if point1 and point2:
                    alignment = 100 - abs(point1[1] - point2[1]) * 200
                    scores.append(max(0, alignment))
        
        return np.mean(scores) if scores else 50.0

    def _calculate_speed_score(self, pose_data: List[Dict]) -> float:
        """Calculate speed/movement score based on frame transitions"""
        if len(pose_data) < 2:
            return 50.0
        
        movement_scores = []
        
        for i in range(1, len(pose_data)):
            prev_landmarks = pose_data[i-1]
            curr_landmarks = pose_data[i]
            
            key_points = [11, 12, 23, 24]
            frame_movement = 0
            
            for point in key_points:
                if point in prev_landmarks and point in curr_landmarks:
                    prev_pos = np.array(prev_landmarks[point])
                    curr_pos = np.array(curr_landmarks[point])
                    movement = np.linalg.norm(curr_pos - prev_pos)
                    frame_movement += movement
            
            movement_scores.append(frame_movement)
        
        avg_movement = np.mean(movement_scores) if movement_scores else 0
        speed_score = min(100, avg_movement * 2000)
        
        return speed_score

    def _generate_performance_metrics(
        self, 
        pose_data: List[Dict], 
        stability_scores: List[float], 
        technique_scores: List[float], 
        sport: str, 
        activity: str
    ) -> PerformanceMetrics:
        """Generate comprehensive performance metrics"""
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0
        avg_technique = np.mean(technique_scores) if technique_scores else 0
        
        form_consistency = max(0, 100 - np.std(technique_scores) * 10) if len(technique_scores) > 1 else 75
        posture_score = (avg_stability + form_consistency) / 2
        speed_score = self._calculate_speed_score(pose_data)
        
        overall_score = (
            avg_technique * 0.35 +
            posture_score * 0.25 +
            avg_stability * 0.20 +
            speed_score * 0.10 +
            form_consistency * 0.10
        )
        
        injury_risk = "Low" if overall_score >= 80 else "Medium" if overall_score >= 60 else "High"
        
        recommendations = self._generate_recommendations(avg_technique, posture_score, avg_stability, sport, activity)
        
        detailed_analysis = {
            'frame_count': len(pose_data),
            'stability_variance': np.std(stability_scores) if stability_scores else 0,
            'technique_variance': np.std(technique_scores) if technique_scores else 0,
            'sport': sport,
            'activity': activity,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return PerformanceMetrics(
            overall_score=round(overall_score, 2),
            technique_score=round(avg_technique, 2),
            posture_score=round(posture_score, 2),
            stability_score=round(avg_stability, 2),
            speed_score=round(speed_score, 2),
            form_consistency=round(form_consistency, 2),
            injury_risk=injury_risk,
            recommendations=recommendations,
            detailed_analysis=detailed_analysis
        )

    def _generate_recommendations(self, technique: float, posture: float, stability: float, sport: str, activity: str) -> List[str]:
        """Generate personalized training recommendations"""
        recommendations = []
        
        if technique < 70:
            recommendations.append(f"Focus on {sport} {activity} technique drills")
            recommendations.append("Practice basic movement patterns slowly")
        
        if posture < 70:
            recommendations.append("Work on core strengthening exercises")
            recommendations.append("Practice posture alignment drills")
        
        if stability < 70:
            recommendations.append("Include balance training in your routine")
            recommendations.append("Focus on single-leg stability exercises")
        
        if sport == 'cricket':
            if activity == 'bowling':
                recommendations.append("Practice bowling action in front of mirror")
                recommendations.append("Work on shoulder flexibility and strength")
        elif sport == 'football':
            recommendations.append("Practice ball control and first touch")
            recommendations.append("Focus on leg strength training")
        elif sport == 'athletics':
            recommendations.append("Work on running form and cadence")
            recommendations.append("Include plyometric exercises")
        
        if not recommendations:
            recommendations.append("Continue practicing to maintain good form")
            recommendations.append("Consider working with a coach for advanced techniques")
        
        return recommendations

# Initialize the analyzer
analyzer = MediaPipePoseAnalyzer()
athletes_db = {}
assessments_db = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_performance():
    try:
        # Get form data
        athlete_name = request.form.get('athleteName')
        athlete_age = int(request.form.get('athleteAge'))
        athlete_sport = request.form.get('athleteSport')
        athlete_activity = request.form.get('athleteActivity')
        athlete_region = request.form.get('athleteRegion')
        athlete_phone = request.form.get('athletePhone', '')
        
        # Check if demo mode
        is_demo = request.form.get('isDemo') == 'true'
        
        if is_demo:
            # Return mock data for demo
            return jsonify({
                'success': True,
                'overall_score': 78.5,
                'detailed_scores': {
                    'technique': 75.2,
                    'posture': 82.1,
                    'stability': 79.8,
                    'speed': 68.5,
                    'form_consistency': 85.2
                },
                'injury_risk': 'Low',
                'recommendations': [
                    f'Focus on {athlete_sport} {athlete_activity} technique drills',
                    'Work on core strengthening exercises',
                    'Continue practicing to maintain good form'
                ]
            })
        
        # Handle file upload
        if 'videoFile' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        
        file = request.files['videoFile']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Register athlete
            athlete_id = f"ATH_{timestamp}"
            athlete = AthleteProfile(
                athlete_id=athlete_id,
                name=athlete_name,
                age=athlete_age,
                sport=athlete_sport,
                region=athlete_region,
                phone=athlete_phone,
                created_at=datetime.now()
            )
            athletes_db[athlete_id] = athlete
            
            # Analyze video
            metrics = analyzer.analyze_video(filepath, athlete_sport, athlete_activity)
            
            # Store assessment
            assessment_id = f"ASSESS_{timestamp}"
            assessments_db[assessment_id] = {
                'athlete_id': athlete_id,
                'assessment_id': assessment_id,
                'video_path': filepath,
                'sport': athlete_sport,
                'activity': athlete_activity,
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            
            # Clean up uploaded file after analysis
            try:
                os.remove(filepath)
            except:
                pass  # File cleanup is not critical
            
            # Return results
            return jsonify({
                'success': True,
                'assessment_id': assessment_id,
                'athlete_id': athlete_id,
                'overall_score': metrics.overall_score,
                'detailed_scores': {
                    'technique': metrics.technique_score,
                    'posture': metrics.posture_score,
                    'stability': metrics.stability_score,
                    'speed': metrics.speed_score,
                    'form_consistency': metrics.form_consistency
                },
                'injury_risk': metrics.injury_risk,
                'recommendations': metrics.recommendations
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/athlete/<athlete_id>')
def get_athlete(athlete_id):
    if athlete_id not in athletes_db:
        return jsonify({'error': 'Athlete not found'}), 404
    
    athlete = athletes_db[athlete_id]
    assessments = [
        assessment for assessment in assessments_db.values() 
        if assessment['athlete_id'] == athlete_id
    ]
    
    return jsonify({
        'athlete_profile': {
            'id': athlete.athlete_id,
            'name': athlete.name,
            'age': athlete.age,
            'sport': athlete.sport,
            'region': athlete.region,
            'registered_at': athlete.created_at.isoformat()
        },
        'assessment_history': [
            {
                'assessment_id': a['assessment_id'],
                'sport': a['sport'],
                'activity': a['activity'],
                'overall_score': a['metrics'].overall_score,
                'date': a['timestamp'].isoformat()
            } for a in assessments
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)