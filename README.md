# Sports AI Analysis - Setup Instructions

This is a complete working Sports AI Analysis application that uses computer vision and MediaPipe to analyze athlete performance from video uploads.

## Project Structure

```
sports-ai-analysis/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend HTML file
├── uploads/              # Directory for video uploads (auto-created)
└── README.md             # Setup instructions
```

## Prerequisites

- Python 3.10
- pip (Python package manager)
- At least 4GB RAM (for MediaPipe processing)
- Webcam or video files for testing

## Installation Steps

### 1. Clone/Create Project Directory

```bash
mkdir sports-ai-analysis
cd sports-ai-analysis
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
py -3.10 -m venv venv310  
.\venv310\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Directory Structure

```bash
mkdir templates
mkdir uploads
```

### 5. Add the Files

- Place `app.py` in the root directory
- Place `index.html` in the `templates/` directory
- Place `requirements.txt` in the root directory

## Running the Application

1. **Start the Flask Server:**
   ```bash
   python app.py
   ```

2. **Open Your Browser:**
   Navigate to `http://localhost:5000`

3. **Test the Application:**
   - Fill in athlete details (name, age, sport, etc.)
   - Either upload a video file or click "Run Demo" for sample data
   - Click "Get Started" to analyze performance

## Features

### Frontend Features
- Responsive mobile-first design
- Drag-and-drop video upload
- Form validation
- Real-time greeting updates
- Loading animations
- Error handling
- Demo mode for testing

### Backend Features
- Video processing with OpenCV
- Pose estimation using MediaPipe
- Sport-specific analysis (Cricket, Football, Athletics)
- Performance metrics calculation
- RESTful API endpoints
- File upload handling
- Comprehensive error handling

### AI Analysis Capabilities
- **Technique Analysis**: Sport-specific movement analysis
- **Posture Assessment**: Body alignment and positioning
- **Stability Scoring**: Balance and core stability
- **Speed Analysis**: Movement dynamics
- **Form Consistency**: Movement pattern consistency
- **Injury Risk Assessment**: Based on technique and stability
- **Personalized Recommendations**: Tailored improvement suggestions

## API Endpoints

### POST /api/analyze
Analyze athlete performance from uploaded video or demo mode.

**Form Data:**
- `athleteName`: Athlete's name
- `athleteAge`: Age (10-30)
- `athleteSport`: Sport (cricket/football/athletics)
- `athleteActivity`: Activity (bowling/batting/fielding/dribbling/shooting/etc.)
- `athleteRegion`: State/UT
- `athletePhone`: Phone number (optional)
- `videoFile`: Video file (if not demo mode)
- `isDemo`: Boolean for demo mode

**Response:**
```json
{
  "success": true,
  "assessment_id": "ASSESS_20241216120000",
  "athlete_id": "ATH_20241216120000",
  "overall_score": 78.5,
  "detailed_scores": {
    "technique": 75.2,
    "posture": 82.1,
    "stability": 79.8,
    "speed": 68.5,
    "form_consistency": 85.2
  },
  "injury_risk": "Low",
  "recommendations": [
    "Focus on cricket bowling technique drills",
    "Work on core strengthening exercises",
    "Continue practicing to maintain good form"
  ]
}
```

### GET /api/athlete/{athlete_id}
Get athlete profile and assessment history.

## Supported Sports & Activities

### Cricket
- Bowling
- Batting  
- Fielding

### Football
- Dribbling
- Shooting
- Goalkeeping

### Athletics
- Running
- Jumping
- Throwing

## Video Requirements

- **Format**: MP4, AVI, MOV, MKV, WebM
- **Size**: Maximum 100MB
- **Quality**: Clear visibility of athlete's body
- **Duration**: 5-60 seconds recommended
- **Content**: Single athlete performing the activity

## Troubleshooting

### Common Issues

1. **MediaPipe Installation Error:**
   ```bash
   pip install --upgrade pip
   pip install mediapipe --no-cache-dir
   ```

2. **OpenCV Issues on macOS:**
   ```bash
   brew install opencv
   pip install opencv-python
   ```

3. **Port Already in Use:**
   Change the port in `app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

4. **File Upload Errors:**
   - Check file size (max 100MB)
   - Ensure supported video format
   - Verify upload directory permissions

### Performance Optimization

1. **For Better Analysis:**
   - Use well-lit videos
   - Ensure full body is visible
   - Avoid shaky camera movement
   - Record from side angle for best pose detection

2. **For Faster Processing:**
   - Use shorter video clips (5-30 seconds)
   - Reduce video resolution if needed
   - Close other applications to free up RAM

## Development Notes

### Architecture
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Backend**: Flask with RESTful API design
- **AI Processing**: MediaPipe for pose estimation
- **Video Processing**: OpenCV for frame handling

### Key Files
- `app.py`: Main Flask application with AI analysis logic
- `templates/index.html`: Complete frontend with modern UI
- `requirements.txt`: All necessary Python dependencies

### Extending the Application

1. **Adding New Sports:**
   - Update `sport_keypoints` in `MediaPipePoseAnalyzer`
   - Add sport-specific analysis methods
   - Update frontend sport options

2. **Adding New Metrics:**
   - Modify `_generate_performance_metrics()` method
   - Update `PerformanceMetrics` dataclass
   - Add corresponding UI elements

3. **Database Integration:**
   - Replace in-memory storage with proper database
   - Add user authentication
   - Implement assessment history

## Security Considerations

- File upload validation implemented
- File size limits enforced
- Temporary file cleanup after processing
- CORS configuration for development

## Production Deployment

For production deployment:
1. Use a production WSGI server (Gunicorn)
2. Configure proper database (PostgreSQL/MySQL)
3. Implement user authentication
4. Set up proper file storage (cloud storage)
5. Add monitoring and logging
6. Configure environment variables

## Support

This application provides a complete working implementation of sports performance analysis using computer vision. The demo mode allows testing without video uploads, and the real video analysis provides meaningful insights for athlete development.

For technical issues or questions, review the error messages in the browser console and server logs.