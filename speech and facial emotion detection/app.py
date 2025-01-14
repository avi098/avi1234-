import os
import shutil
import subprocess
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import soundfile as sf
import speech_recognition as sr
import logging
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import threading
from queue import Queue
import json
import google.generativeai as genai
from datetime import datetime
import requests
from pydub import AudioSegment
from pathlib import Path
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionDetector:
    """Handles real-time facial emotion detection using OpenCV and a pre-trained model."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = self.load_emotion_model()
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_history = []
        self.history_size = 5
        self.cap = None
        self.last_emotion = "Unknown"
        self.emotion_confidence = 0.0
        self.running = True
        self.initialize_camera()

    def initialize_camera(self):
        """Initialize camera with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    logger.info("Camera initialized successfully")
                    break
            except Exception as e:
                logger.error(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if self.cap:
                    self.cap.release()
                time.sleep(1)
        
        if not self.cap or not self.cap.isOpened():
            logger.error("Failed to initialize camera after all attempts")
            self.running = False

    def load_emotion_model(self) -> Optional[Any]:
        """Load the pre-trained emotion detection model."""
        try:
            model_path = Path('face_emotion_model.h5')
            if not model_path.exists():
                logger.error("Emotion model file not found")
                return None
            return load_model(str(model_path))
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            return None

    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for emotion detection."""
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0
        return face_img

    def get_smooth_emotion(self, emotion_pred: np.ndarray) -> Tuple[str, float]:
        """Apply smoothing to emotion predictions using historical data."""
        self.emotion_history.append(emotion_pred)
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
        
        avg_pred = np.mean(self.emotion_history, axis=0)
        emotion_idx = np.argmax(avg_pred)
        confidence = float(avg_pred[emotion_idx])
        return self.emotion_labels[emotion_idx], confidence

    def detect_emotion(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Detect emotion from frame with confidence score."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        max_face_area = 0
        main_face_emotion = self.last_emotion
        confidence = self.emotion_confidence
        
        for (x, y, w, h) in faces:
            face_area = w * h
            if face_area > max_face_area:
                max_face_area = face_area
                roi = frame[y:y+h, x:x+w]
                processed_face = self.preprocess_face(roi)
                
                if self.emotion_model is not None:
                    emotion_pred = self.emotion_model.predict(processed_face)[0]
                    emotion_label, conf = self.get_smooth_emotion(emotion_pred)
                    main_face_emotion = emotion_label
                    confidence = conf
                    
                    # Draw rectangle and emotion label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"{emotion_label} ({conf:.2f})"
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        self.last_emotion = main_face_emotion
        self.emotion_confidence = confidence
        return frame, main_face_emotion, confidence

    def generate_frames(self):
        """Generate video frames with emotion detection."""
        while self.running:
            success, frame = self.cap.read()
            if not success:
                break
                
            frame, _, _ = self.detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def cleanup(self):
        """Release resources."""
        self.running = False
        if self.cap is not None:
            self.cap.release()

class SpeechEmotionDetector:
    """Handles speech emotion detection using a pre-trained model."""
    
    def __init__(self):
        self.model = self.load_model()
        self.emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
        
    def load_model(self) -> Optional[Any]:
        """Load the pre-trained speech emotion model."""
        try:
            model_path = Path('speech_emotion_model.h5')
            if not model_path.exists():
                logger.error("Speech emotion model file not found")
                return None
            return load_model(str(model_path))
        except Exception as e:
            logger.error(f"Error loading speech emotion model: {e}")
            return None

    def extract_features(self, audio_path: Path) -> np.ndarray:
        """Extract MFCC features from audio file."""
        try:
            y, sr = librosa.load(audio_path, duration=3, offset=0.5)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            return np.expand_dims(mfcc_scaled, axis=(0, -1))
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return None

    def predict_emotion(self, audio_path: Path) -> Tuple[Optional[str], float]:
        """Predict emotion from audio file."""
        if self.model is None:
            return None, 0.0
            
        features = self.extract_features(audio_path)
        if features is None:
            return None, 0.0
            
        try:
            predictions = self.model.predict(features)[0]
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[emotion_idx])
            return self.emotion_labels[emotion_idx], confidence
        except Exception as e:
            logger.error(f"Error predicting speech emotion: {e}")
            return None, 0.0

class EmotionFusion:
    """Combines facial and speech emotion predictions."""
    
    @staticmethod
    def fuse_emotions(face_emotion: str, face_confidence: float,
                     speech_emotion: str, speech_confidence: float) -> Dict[str, Any]:
        """Fuse facial and speech emotions with confidence weighting."""
        if face_emotion == "Unknown" and speech_emotion is None:
            return {
                "primary_emotion": "Unknown",
                "confidence": 0.0,
                "face_emotion": face_emotion,
                "speech_emotion": speech_emotion or "Unknown"
            }
        
        # Normalize confidences
        total_confidence = face_confidence + speech_confidence
        if total_confidence == 0:
            face_weight = 0.5
            speech_weight = 0.5
        else:
            face_weight = face_confidence / total_confidence
            speech_weight = speech_confidence / total_confidence
        
        # If emotions match, use that emotion with combined confidence
        if face_emotion == speech_emotion:
            return {
                "primary_emotion": face_emotion,
                "confidence": total_confidence / 2,
                "face_emotion": face_emotion,
                "speech_emotion": speech_emotion
            }
        
        # If emotions differ, use the one with higher confidence
        if face_confidence > speech_confidence:
            primary_emotion = face_emotion
            confidence = face_confidence
        else:
            primary_emotion = speech_emotion
            confidence = speech_confidence
            
        return {
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "face_emotion": face_emotion,
            "speech_emotion": speech_emotion
        }

class PsychiatristBot:
    """Handles interaction with Gemini AI for psychiatric responses."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.conversation_history = []
        self.init_psychiatric_context()
    
    def init_psychiatric_context(self):
        """Initialize the psychiatric context for the Gemini model."""
        context = """You are an experienced, empathetic psychiatrist with expertise in emotional wellness 
        and mental health. When responding to patients:
        1. Consider both facial expressions and speech tone emotions
        2. Acknowledge emotional state with confidence levels
        3. Provide supportive, professional feedback
        4. Ask relevant follow-up questions when needed
        5. Maintain a warm, professional tone
        6. Offer specific coping strategies when appropriate
        7. Note any emotional discrepancies between face and speech
        Keep responses concise but meaningful."""
        
        self.conversation_history = [{"role": "system", "content": context}]
    
    def analyze_and_respond(self, 
                          transcript: str,
                          emotion_data: Dict[str, Any],
                          session_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate a psychiatric response based on speech and emotion data."""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create detailed prompt including emotion confidence levels
            analysis_prompt = f"""
            Time: {current_time}
            Primary Detected Emotion: {emotion_data['primary_emotion']} (Confidence: {emotion_data['confidence']:.2f})
            Facial Expression: {emotion_data['face_emotion']}
            Speech Emotion: {emotion_data['speech_emotion']}
            Patient's Statement: {transcript}
            
            Based on the detected emotions and statement, provide a professional psychiatric response.
            Consider any discrepancy between facial and speech emotions in your analysis.
            """
            
            if session_history:
                self.conversation_history.extend(session_history)
            
            response = self.model.generate_content(analysis_prompt)
            psychiatric_response = response.text
            
            # Update conversation history with detailed emotion data
            self.conversation_history.extend([
                {
                    "role": "user",
                    "content": transcript,
                    "emotion_data": emotion_data,
                    "timestamp": current_time
                },
                {
                    "role": "assistant",
                    "content": psychiatric_response,
                    "timestamp": current_time
                }
            ])
            
            return {
                "response": psychiatric_response,
                "emotion_data": emotion_data,
                "success": True,
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Error generating psychiatric response: {e}")
            return {
                "error": "Failed to generate psychiatric response",
                "success": False
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """Return the conversation history."""
        return self.conversation_history

class SpeechProcessor:
    """Enhanced speech recognition with multiple engines and fallback options."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engines = ['google', 'sphinx']  # Added offline fallback
        self.current_engine = 'google'
    
    def optimize_audio(self, audio_data):
        """Optimize audio for better recognition."""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Apply noise reduction
            audio_data = librosa.effects.preemphasis(audio_data)
            
            return audio_data
        except Exception as e:
            logger.error(f"Audio optimization error: {e}")
            return audio_data

    def transcribe_audio(self, audio_path: Path) -> Tuple[str, bool]:
        """Transcribe audio with fallback options."""
        for engine in self.engines:
            try:
                with sr.AudioFile(str(audio_path)) as source:
                    audio = self.recognizer.record(source)
                    
                    if engine == 'google':
                        try:
                            text = self.recognizer.recognize_google(audio)
                            return text, True
                        except requests.exceptions.RequestException:
                            logger.warning("Google Speech API connection failed, trying next engine")
                            continue
                    elif engine == 'sphinx':
                        try:
                            text = self.recognizer.recognize_sphinx(audio)
                            return text, True
                        except sr.UnknownValueError:
                            continue
                        
            except Exception as e:
                logger.error(f"Error with {engine} recognition: {e}")
                continue
                
        return "Speech recognition failed", False


class AudioProcessor:
    """Handles audio processing and speech-to-text conversion."""
    
    def __init__(self, psychiatrist_bot):
        self.speech_processor = SpeechProcessor()
        self.recognizer = sr.Recognizer()  # Initialize the recognizer here
        self.emotion_detector = None
        self.speech_emotion_detector = SpeechEmotionDetector()
        self.psychiatrist_bot = psychiatrist_bot
        self.emotion_fusion = EmotionFusion()
        
        # Optimize speech recognition settings after initializing recognizer
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5

    def set_emotion_detector(self, detector: EmotionDetector):
        """Set the emotion detector reference."""
        self.emotion_detector = detector
    
    def process_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Process audio with enhanced error handling and optimization."""
        try:
            # Convert to WAV with error handling
            wav_path = audio_path.with_suffix('.wav')
            if not AudioUtils.convert_audio_to_wav(audio_path, wav_path):
                return {
                    'error': 'Audio conversion failed',
                    'success': False
                }

            # Get transcript and speech emotion
            transcript, speech_success = self.speech_processor.transcribe_audio(wav_path)
            speech_emotion, speech_confidence = self.speech_emotion_detector.predict_emotion(wav_path)

            # Get current facial emotion
            face_emotion = self.emotion_detector.last_emotion if self.emotion_detector else "Unknown"
            face_confidence = self.emotion_detector.emotion_confidence if self.emotion_detector else 0.0

            # Fuse emotions
            emotion_data = self.emotion_fusion.fuse_emotions(
                face_emotion, face_confidence,
                speech_emotion, speech_confidence
            )

            # Get psychiatric response
            response_data = self.psychiatrist_bot.analyze_and_respond(
                transcript,
                emotion_data
            )

            return {
                'transcript': transcript,
                'emotion_data': emotion_data,
                'psychiatric_response': response_data['response'],
                'success': True
            }

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {
                'error': str(e),
                'success': False
            }
        finally:
            # Cleanup temporary files
            try:
                if wav_path.exists():
                    wav_path.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")

    def process_transcript_and_emotions(self, transcript: str, wav_path: Path) -> Dict[str, Any]:
        """Process transcript and emotions from audio."""
        try:
            # Get speech emotion
            speech_emotion, speech_confidence = self.speech_emotion_detector.predict_emotion(wav_path)

            # Get current facial emotion
            face_emotion = self.emotion_detector.last_emotion if self.emotion_detector else "Unknown"
            face_confidence = self.emotion_detector.emotion_confidence if self.emotion_detector else 0.0

            # Fuse emotions
            emotion_data = self.emotion_fusion.fuse_emotions(
                face_emotion, face_confidence,
                speech_emotion, speech_confidence
            )

            # Get psychiatric response
            response_data = self.psychiatrist_bot.analyze_and_respond(
                transcript,
                emotion_data
            )

            return {
                'transcript': transcript,
                'emotion_data': emotion_data,
                'psychiatric_response': response_data['response'],
                'success': True
            }

        except Exception as e:
            logger.error(f"Error processing transcript and emotions: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def convert_audio_to_wav(self, input_path: Path, output_path: Path) -> bool:
        """Convert audio to WAV with enhanced error handling."""
        try:
            audio = AudioSegment.from_file(str(input_path))
            audio.export(str(output_path), format='wav')
            return True
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return False

class AudioUtils:
    """Utility functions for audio processing."""
    
    @staticmethod
    def convert_audio_to_wav(input_path: Path, output_path: Path) -> bool:
        """Convert audio file to WAV format."""
        try:
            data, sr = librosa.load(str(input_path), sr=None)  # Preserve original sample rate
            sf.write(str(output_path), data, sr, subtype='PCM_16')
            return True
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return False

    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data."""
        return librosa.util.normalize(audio_data)

class Config:
    """Application configuration."""
    # Base paths
    BASE_DIR = Path(os.getcwd())
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # API Configuration
    GEMINI_API_KEY = "AIzaSyCR0HvEJwG73o--cJ9rHZPjcDBHMqrFJIw"  # Replace with your actual API key
    
    # Model paths
    FACE_EMOTION_MODEL_PATH = BASE_DIR / 'face_emotion_model.h5'
    SPEECH_EMOTION_MODEL_PATH = BASE_DIR / 'speech_emotion_model.h5'
    
    # Emotion detection settings
    EMOTION_CONFIDENCE_THRESHOLD = 0.6
    EMOTION_HISTORY_SIZE = 5
    
    # Audio processing settings
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 5  # seconds
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        try:
            # Check if API key is set
            if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
                logger.error("Please set your Gemini API key in the Config class")
                return False
            
            # Create required directories
            cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
            (cls.BASE_DIR / 'models').mkdir(parents=True, exist_ok=True)
            
            # Validate model files
            if not cls.FACE_EMOTION_MODEL_PATH.exists():
                logger.error(f"Face emotion model not found at {cls.FACE_EMOTION_MODEL_PATH}")
                return False
            
            if not cls.SPEECH_EMOTION_MODEL_PATH.exists():
                logger.error(f"Speech emotion model not found at {cls.SPEECH_EMOTION_MODEL_PATH}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    @classmethod
    def init_app(cls, app):
        """Initialize Flask application with config values."""
        app.config['UPLOAD_FOLDER'] = str(cls.UPLOAD_FOLDER)
        app.config['MAX_CONTENT_LENGTH'] = cls.MAX_CONTENT_LENGTH

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure upload folder and other settings
app.config['UPLOAD_FOLDER'] = str(Path(os.getcwd()) / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.urandom(24)  # Add secure secret key

# Create required directories
Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Validate configuration
if not Config.validate_config():
    raise RuntimeError("Invalid configuration. Please check the logs for details.")

# Function to initialize application components
def initialize_components():
    """Initialize all application components with proper error handling."""
    try:
        # Validate configuration first
        if not Config.validate_config():
            raise RuntimeError("Invalid configuration. Please check the logs for details.")
        
        # Initialize components in order
        psychiatrist_bot = PsychiatristBot(api_key=Config.GEMINI_API_KEY)
        emotion_detector = EmotionDetector()
        
        # Initialize audio processor with psychiatrist bot
        audio_processor = AudioProcessor(psychiatrist_bot)
        
        # Set emotion detector reference in audio processor
        audio_processor.set_emotion_detector(emotion_detector)
        
        logger.info("All components initialized successfully")
        return psychiatrist_bot, emotion_detector, audio_processor
    
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


# Initialize components
try:
    psychiatrist_bot, emotion_detector, audio_processor = initialize_components()
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    raise

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        emotion_detector.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/upload', methods=['POST'])
def upload():
    """Enhanced file upload handler with better error handling."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Empty filename'}), 400

        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Process file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        
        try:
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save and process file
            file.save(str(filepath))
            results = audio_processor.process_audio(filepath)
            
            if results.get('success', False):
                return jsonify({
                    'transcript': results.get('transcript', ''),
                    'emotion_data': results.get('emotion_data', {}),
                    'psychiatric_response': results.get('psychiatric_response', ''),
                    'success': True
                })
            else:
                return jsonify({'error': results.get('error', 'Unknown error')}), 500

        finally:
            # Clean up uploaded file
            if filepath.exists():
                try:
                    filepath.unlink()
                except Exception as e:
                    logger.error(f"Error removing upload file: {e}")

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/psychiatric_history')
def get_psychiatric_history():
    """Return the conversation history."""
    try:
        history = psychiatrist_bot.get_conversation_history()
        return jsonify({
            'history': history,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error fetching psychiatric history: {e}")
        return jsonify({
            'error': 'Failed to fetch history',
            'status': 'error'
        }), 500

@app.route('/emotion_status')
def get_emotion_status():
    """Get current emotion detection status."""
    try:
        return jsonify({
            'face_emotion': emotion_detector.last_emotion,
            'face_confidence': emotion_detector.emotion_confidence,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error getting emotion status: {e}")
        return jsonify({
            'error': 'Failed to get emotion status',
            'status': 'error'
        }), 500

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera feed."""
    try:
        emotion_detector.cleanup()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return jsonify({
            'error': 'Failed to stop camera',
            'status': 'error'
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large',
        'max_size': app.config['MAX_CONTENT_LENGTH']
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

# Application initialization
if __name__ == '__main__':
    try:
        # Initialize components with error handling
        psychiatrist_bot, emotion_detector, audio_processor = initialize_components()

        # Create required directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Start server with improved settings
        app.run(
            debug=False,
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Application startup error: {e}")
    finally:
        if 'emotion_detector' in locals() and emotion_detector:
            emotion_detector.cleanup()