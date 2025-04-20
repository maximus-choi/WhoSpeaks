from PyQt6.QtWidgets import (QApplication, QTextEdit, QMainWindow, QLabel, QVBoxLayout, QWidget, 
                            QHBoxLayout, QPushButton, QSizePolicy, QGroupBox, QSlider, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QEvent, QTimer
from scipy.spatial.distance import cosine
from RealtimeSTT import AudioToTextRecorder
import numpy as np
import soundcard as sc
import queue
import torch
import time
import sys
import os
import urllib.request
import torchaudio

# Simplified configuration parameters
SILENCE_THRESHS = [0, 0.4]
FINAL_TRANSCRIPTION_MODEL = "distil-large-v3"
FINAL_BEAM_SIZE = 5
REALTIME_TRANSCRIPTION_MODEL = "distil-small.en"
REALTIME_BEAM_SIZE = 5
TRANSCRIPTION_LANGUAGE = "en" # Accuracy in languages ​​other than English is very low.
SILERO_SENSITIVITY = 0.4
WEBRTC_SENSITIVITY = 3
MIN_LENGTH_OF_RECORDING = 0.7
PRE_RECORDING_BUFFER_DURATION = 0.35

# Speaker change detection parameters
DEFAULT_CHANGE_THRESHOLD = 0.7  # Threshold for detecting speaker change
EMBEDDING_HISTORY_SIZE = 5  # Number of embeddings to keep for comparison
MIN_SEGMENT_DURATION = 1.0  # Minimum duration before considering a speaker change
DEFAULT_MAX_SPEAKERS = 4  # Default maximum number of speakers
ABSOLUTE_MAX_SPEAKERS = 10  # Absolute maximum number of speakers allowed

# Global variables
FAST_SENTENCE_END = True
USE_MICROPHONE = False
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
CHANNELS = 1

# Speaker colors - now we have colors for up to 10 speakers
SPEAKER_COLORS = [
    "#FFFF00",  # Yellow
    "#FF0000",  # Red
    "#00FF00",  # Green
    "#00FFFF",  # Cyan
    "#FF00FF",  # Magenta
    "#0000FF",  # Blue
    "#FF8000",  # Orange
    "#00FF80",  # Spring Green
    "#8000FF",  # Purple
    "#FFFFFF",  # White
]

# Color names for display
SPEAKER_COLOR_NAMES = [
    "Yellow",
    "Red",
    "Green",
    "Cyan",
    "Magenta",
    "Blue",
    "Orange",
    "Spring Green",
    "Purple",
    "White"
]


class SpeechBrainEncoder:
    """ECAPA-TDNN encoder from SpeechBrain for speaker embeddings"""
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.embedding_dim = 192  # ECAPA-TDNN default dimension
        self.model_loaded = False
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _download_model(self):
        """Download pre-trained SpeechBrain ECAPA-TDNN model if not present"""
        model_url = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt"
        model_path = os.path.join(self.cache_dir, "embedding_model.ckpt")
        
        if not os.path.exists(model_path):
            print(f"Downloading ECAPA-TDNN model to {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
        
        return model_path
    
    def load_model(self):
        """Load the ECAPA-TDNN model"""
        try:
            # Import SpeechBrain
            from speechbrain.pretrained import EncoderClassifier
            
            # Get model path
            model_path = self._download_model()
            
            # Load the pre-trained model
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.cache_dir,
                run_opts={"device": self.device}
            )
            
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Error loading ECAPA-TDNN model: {e}")
            return False
    
    def embed_utterance(self, audio, sr=16000):
        """Extract speaker embedding from audio"""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert numpy array to torch tensor
            if isinstance(audio, np.ndarray):
                waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            else:
                waveform = audio.unsqueeze(0)
            
            # Ensure sample rate matches model expected rate
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform)
                
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return np.zeros(self.embedding_dim)


class AudioProcessor:
    """Processes audio data to extract speaker embeddings"""
    def __init__(self, encoder):
        self.encoder = encoder
    
    def extract_embedding(self, audio_int16):
        try:
            # Convert int16 audio data to float32
            float_audio = audio_int16.astype(np.float32) / 32768.0
            
            # Normalize if needed
            if np.abs(float_audio).max() > 1.0:
                float_audio = float_audio / np.abs(float_audio).max()
            
            # Extract embedding using the loaded encoder
            embedding = self.encoder.embed_utterance(float_audio)
            
            return embedding
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return np.zeros(self.encoder.embedding_dim)


class EncoderLoaderThread(QThread):
    """Thread for loading the speaker encoder model"""
    model_loaded = pyqtSignal(object)
    progress_update = pyqtSignal(str)
    
    def run(self):
        try:
            self.progress_update.emit("Initializing speaker encoder model...")
            
            # Check device
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.progress_update.emit(f"Using device: {device_str}")
            
            # Create SpeechBrain encoder
            self.progress_update.emit("Loading ECAPA-TDNN model...")
            encoder = SpeechBrainEncoder(device=device_str)
            
            # Load the model
            success = encoder.load_model()
            
            if success:
                self.progress_update.emit("ECAPA-TDNN model loading complete!")
                self.model_loaded.emit(encoder)
            else:
                self.progress_update.emit("Failed to load ECAPA-TDNN model. Using fallback...")
                self.model_loaded.emit(None)
        except Exception as e:
            self.progress_update.emit(f"Model loading error: {e}")
            self.model_loaded.emit(None)


class SpeakerChangeDetector:
    """Modified speaker change detector that supports a configurable number of speakers"""
    def __init__(self, embedding_dim=192, change_threshold=DEFAULT_CHANGE_THRESHOLD, max_speakers=DEFAULT_MAX_SPEAKERS):
        self.embedding_dim = embedding_dim
        self.change_threshold = change_threshold
        self.max_speakers = min(max_speakers, ABSOLUTE_MAX_SPEAKERS)  # Ensure we don't exceed absolute max
        self.current_speaker = 0  # Initial speaker (0 to max_speakers-1)
        self.previous_embeddings = []
        self.last_change_time = time.time()
        self.mean_embeddings = [None] * self.max_speakers  # Mean embeddings for each speaker
        self.speaker_embeddings = [[] for _ in range(self.max_speakers)]  # All embeddings for each speaker
        self.last_similarity = 0.0
        self.active_speakers = set([0])  # Track which speakers have been detected
        
    def set_max_speakers(self, max_speakers):
        """Update the maximum number of speakers"""
        new_max = min(max_speakers, ABSOLUTE_MAX_SPEAKERS)
        
        # If reducing the number of speakers
        if new_max < self.max_speakers:
            # Remove any speakers beyond the new max
            for speaker_id in list(self.active_speakers):
                if speaker_id >= new_max:
                    self.active_speakers.discard(speaker_id)
            
            # Ensure current speaker is valid
            if self.current_speaker >= new_max:
                self.current_speaker = 0
        
        # Expand arrays if increasing max speakers
        if new_max > self.max_speakers:
            # Extend mean_embeddings array
            self.mean_embeddings.extend([None] * (new_max - self.max_speakers))
            
            # Extend speaker_embeddings array
            self.speaker_embeddings.extend([[] for _ in range(new_max - self.max_speakers)])
        
        # Truncate arrays if decreasing max speakers
        else:
            self.mean_embeddings = self.mean_embeddings[:new_max]
            self.speaker_embeddings = self.speaker_embeddings[:new_max]
        
        self.max_speakers = new_max
        
    def set_change_threshold(self, threshold):
        """Update the threshold for detecting speaker changes"""
        self.change_threshold = max(0.1, min(threshold, 0.99))
        
    def add_embedding(self, embedding, timestamp=None):
        """Add a new embedding and check if there's a speaker change"""
        current_time = timestamp or time.time()
        
        # Initialize first speaker if no embeddings yet
        if not self.previous_embeddings:
            self.previous_embeddings.append(embedding)
            self.speaker_embeddings[self.current_speaker].append(embedding)
            if self.mean_embeddings[self.current_speaker] is None:
                self.mean_embeddings[self.current_speaker] = embedding.copy()
            return self.current_speaker, 1.0
        
        # Calculate similarity with current speaker's mean embedding
        current_mean = self.mean_embeddings[self.current_speaker]
        if current_mean is not None:
            similarity = 1.0 - cosine(embedding, current_mean)
        else:
            # If no mean yet, compare with most recent embedding
            similarity = 1.0 - cosine(embedding, self.previous_embeddings[-1])
        
        self.last_similarity = similarity
        
        # Decide if this is a speaker change
        time_since_last_change = current_time - self.last_change_time
        is_speaker_change = False
        
        # Only consider change if minimum time has passed since last change
        if time_since_last_change >= MIN_SEGMENT_DURATION:
            # Check similarity against threshold
            if similarity < self.change_threshold:
                # Compare with all other speakers' means if available
                best_speaker = self.current_speaker
                best_similarity = similarity
                
                # Check each active speaker
                for speaker_id in range(self.max_speakers):
                    if speaker_id == self.current_speaker:
                        continue
                        
                    speaker_mean = self.mean_embeddings[speaker_id]
                    
                    if speaker_mean is not None:
                        # Calculate similarity with this speaker
                        speaker_similarity = 1.0 - cosine(embedding, speaker_mean)
                        
                        # If more similar to this speaker, update best match
                        if speaker_similarity > best_similarity:
                            best_similarity = speaker_similarity
                            best_speaker = speaker_id
                
                # If best match is different from current speaker, change speaker
                if best_speaker != self.current_speaker:
                    is_speaker_change = True
                    self.current_speaker = best_speaker
                # If no good match with existing speakers and we haven't used all speakers yet
                elif len(self.active_speakers) < self.max_speakers:
                    # Find the next unused speaker ID
                    for new_id in range(self.max_speakers):
                        if new_id not in self.active_speakers:
                            is_speaker_change = True
                            self.current_speaker = new_id
                            self.active_speakers.add(new_id)
                            break
        
        # Handle speaker change
        if is_speaker_change:
            self.last_change_time = current_time
        
        # Update embeddings
        self.previous_embeddings.append(embedding)
        if len(self.previous_embeddings) > EMBEDDING_HISTORY_SIZE:
            self.previous_embeddings.pop(0)
        
        # Update current speaker's embeddings and mean
        self.speaker_embeddings[self.current_speaker].append(embedding)
        self.active_speakers.add(self.current_speaker)
        
        if len(self.speaker_embeddings[self.current_speaker]) > 30:  # Limit history size
            self.speaker_embeddings[self.current_speaker] = self.speaker_embeddings[self.current_speaker][-30:]
            
        # Update mean embedding for current speaker
        if self.speaker_embeddings[self.current_speaker]:
            self.mean_embeddings[self.current_speaker] = np.mean(
                self.speaker_embeddings[self.current_speaker], axis=0
            )
        
        return self.current_speaker, similarity
    
    def get_color_for_speaker(self, speaker_id):
        """Return color for speaker ID (0 to max_speakers-1)"""
        if 0 <= speaker_id < len(SPEAKER_COLORS):
            return SPEAKER_COLORS[speaker_id]
        return "#FFFFFF"  # Default to white if out of range
    
    def get_status_info(self):
        """Return status information about the speaker change detector"""
        speaker_counts = [len(self.speaker_embeddings[i]) for i in range(self.max_speakers)]
        
        return {
            "current_speaker": self.current_speaker,
            "speaker_counts": speaker_counts,
            "active_speakers": len(self.active_speakers),
            "max_speakers": self.max_speakers,
            "last_similarity": self.last_similarity,
            "threshold": self.change_threshold
        }


class TextUpdateThread(QThread):
    text_update_signal = pyqtSignal(str)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        self.text_update_signal.emit(self.text)


class SentenceWorker(QThread):
    sentence_update_signal = pyqtSignal(list, list)
    status_signal = pyqtSignal(str)

    def __init__(self, queue, encoder, change_threshold=DEFAULT_CHANGE_THRESHOLD, max_speakers=DEFAULT_MAX_SPEAKERS):
        super().__init__()
        self.queue = queue
        self.encoder = encoder
        self._is_running = True
        self.full_sentences = []
        self.sentence_speakers = []
        self.change_threshold = change_threshold
        self.max_speakers = max_speakers
        
        # Initialize audio processor for embedding extraction
        self.audio_processor = AudioProcessor(self.encoder)
        
        # Initialize speaker change detector
        self.speaker_detector = SpeakerChangeDetector(
            embedding_dim=self.encoder.embedding_dim,
            change_threshold=self.change_threshold,
            max_speakers=self.max_speakers
        )
        
        # Setup monitoring timer
        self.monitoring_timer = QTimer()
        self.monitoring_timer.timeout.connect(self.report_status)
        self.monitoring_timer.start(2000)  # Report every 2 seconds
    
    def set_change_threshold(self, threshold):
        """Update change detection threshold"""
        self.change_threshold = threshold
        self.speaker_detector.set_change_threshold(threshold)
        
    def set_max_speakers(self, max_speakers):
        """Update maximum number of speakers"""
        self.max_speakers = max_speakers
        self.speaker_detector.set_max_speakers(max_speakers)
    
    def run(self):
        """Main worker thread loop"""
        while self._is_running:
            try:
                text, bytes = self.queue.get(timeout=1)
                self.process_item(text, bytes)
            except queue.Empty:
                continue
    
    def report_status(self):
        """Report status information"""
        # Get status information from speaker detector
        status = self.speaker_detector.get_status_info()
        
        # Prepare status message with information for all speakers
        status_text = f"Current speaker: {status['current_speaker'] + 1}\n"
        status_text += f"Active speakers: {status['active_speakers']} of {status['max_speakers']}\n"
        
        # Show segment counts for each speaker
        for i in range(status['max_speakers']):
            if i < len(SPEAKER_COLOR_NAMES):
                color_name = SPEAKER_COLOR_NAMES[i]
            else:
                color_name = f"Speaker {i+1}"
            status_text += f"Speaker {i+1} ({color_name}) segments: {status['speaker_counts'][i]}\n"
        
        status_text += f"Last similarity score: {status['last_similarity']:.3f}\n"
        status_text += f"Change threshold: {status['threshold']:.2f}\n"
        status_text += f"Total sentences: {len(self.full_sentences)}"
        
        # Send to UI
        self.status_signal.emit(status_text)
    
    def process_item(self, text, bytes):
        """Process a new text-audio pair"""
        # Convert audio data to int16
        audio_int16 = np.int16(bytes * 32767)
        
        # Extract speaker embedding
        speaker_embedding = self.audio_processor.extract_embedding(audio_int16)
        
        # Store sentence and embedding
        self.full_sentences.append((text, speaker_embedding))
        
        # Fill in any missing speaker assignments
        if len(self.sentence_speakers) < len(self.full_sentences) - 1:
            while len(self.sentence_speakers) < len(self.full_sentences) - 1:
                self.sentence_speakers.append(0)  # Default to first speaker
        
        # Detect speaker changes
        speaker_id, similarity = self.speaker_detector.add_embedding(speaker_embedding)
        self.sentence_speakers.append(speaker_id)
        
        # Send updated data to UI
        self.sentence_update_signal.emit(self.full_sentences, self.sentence_speakers)
    
    def stop(self):
        """Stop the worker thread"""
        self._is_running = False
        if self.monitoring_timer.isActive():
            self.monitoring_timer.stop()


class RecordingThread(QThread):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self._is_running = True
        
        # Determine input source
        if USE_MICROPHONE:
            self.device_id = str(sc.default_microphone().name)
            self.include_loopback = False
        else:
            self.device_id = str(sc.default_speaker().name)
            self.include_loopback = True

    def updateDevice(self, device_id, include_loopback):
        self.device_id = device_id
        self.include_loopback = include_loopback

    def run(self):
        while self._is_running:
            try:
                with sc.get_microphone(id=self.device_id, include_loopback=self.include_loopback).recorder(
                    samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE
                ) as mic:
                    # Process audio chunks while device hasn't changed
                    current_device = self.device_id
                    current_loopback = self.include_loopback
                    
                    while self._is_running and current_device == self.device_id and current_loopback == self.include_loopback:
                        # Record audio chunk
                        audio_data = mic.record(numframes=BUFFER_SIZE)
                        
                        # Convert stereo to mono if needed
                        if audio_data.shape[1] > 1 and CHANNELS == 1:
                            audio_data = audio_data[:, 0]
                        
                        # Convert to int16
                        audio_int16 = (audio_data.flatten() * 32767).astype(np.int16)
                        
                        # Feed to recorder
                        audio_bytes = audio_int16.tobytes()
                        self.recorder.feed_audio(audio_bytes)
                    
            except Exception as e:
                print(f"Recording error: {e}")
                # Wait before retry on error
                time.sleep(1)

    def stop(self):
        self._is_running = False


class TextRetrievalThread(QThread):
    textRetrievedFinal = pyqtSignal(str, np.ndarray)
    textRetrievedLive = pyqtSignal(str)
    recorderStarted = pyqtSignal()

    def __init__(self):
        super().__init__()

    def live_text_detected(self, text):
        self.textRetrievedLive.emit(text)

    def run(self):
        recorder_config = {
            'spinner': False,
            'use_microphone': False,
            'model': FINAL_TRANSCRIPTION_MODEL,
            'language': TRANSCRIPTION_LANGUAGE,
            'silero_sensitivity': SILERO_SENSITIVITY,
            'webrtc_sensitivity': WEBRTC_SENSITIVITY,
            'post_speech_silence_duration': SILENCE_THRESHS[1],
            'min_length_of_recording': MIN_LENGTH_OF_RECORDING,
            'pre_recording_buffer_duration': PRE_RECORDING_BUFFER_DURATION,
            'min_gap_between_recordings': 0,
            'enable_realtime_transcription': True,
            'realtime_processing_pause': 0,
            'realtime_model_type': REALTIME_TRANSCRIPTION_MODEL,
            'on_realtime_transcription_update': self.live_text_detected,
            'beam_size': FINAL_BEAM_SIZE,
            'beam_size_realtime': REALTIME_BEAM_SIZE,
            'buffer_size': BUFFER_SIZE,
            'sample_rate': SAMPLE_RATE,
        }

        self.recorder = AudioToTextRecorder(**recorder_config)
        self.recorderStarted.emit()

        def process_text(text):
            bytes = self.recorder.last_transcription_bytes
            self.textRetrievedFinal.emit(text, bytes)

        while True:
            self.recorder.text(process_text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-time Speaker Change Detection")

        self.encoder = None
        self.initialized = False
        self.displayed_text = ""
        self.last_realtime_text = ""
        self.full_sentences = []
        self.sentence_speakers = []
        self.pending_sentences = []
        self.queue = queue.Queue()
        self.recording_thread = None
        self.change_threshold = DEFAULT_CHANGE_THRESHOLD
        self.max_speakers = DEFAULT_MAX_SPEAKERS

        # Create main horizontal layout
        self.mainLayout = QHBoxLayout()

        # Add text edit area to main layout
        self.text_edit = QTextEdit(self)
        self.mainLayout.addWidget(self.text_edit, 1)

        # Create right layout for controls
        self.rightLayout = QVBoxLayout()
        self.rightLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Create all controls
        self.create_controls()

        # Create container for right layout
        self.rightContainer = QWidget()
        self.rightContainer.setLayout(self.rightLayout)
        self.mainLayout.addWidget(self.rightContainer, 0)

        # Set main layout as central widget
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralWidget)

        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #555;
                border-radius: 3px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ddd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QLabel {
                color: #ddd;
            }
            QPushButton {
                background: #444;
                color: #ddd;
                border: 1px solid #555;
                padding: 5px;
                margin-bottom: 10px;
            }
            QPushButton:hover {
                background: #555;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Arial';
                font-size: 16pt;
            }
            QSlider {
                height: 30px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #333;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #666;
                border: 1px solid #777;
                width: 18px;
                margin: -8px 0;
                border-radius: 9px;
            }
        """)

    def create_controls(self):
        # Speaker change threshold control
        self.threshold_group = QGroupBox("Speaker Change Sensitivity")
        threshold_layout = QVBoxLayout()
        
        self.threshold_label = QLabel(f"Change threshold: {self.change_threshold:.2f}")
        threshold_layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(95)
        self.threshold_slider.setValue(int(self.change_threshold * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_explanation = QLabel(
            "If the speakers have similar voices, it would be better to set it above 0.5, and if they have different voices, it would be lower."
        )
        self.threshold_explanation.setWordWrap(True)
        threshold_layout.addWidget(self.threshold_explanation)
        
        self.threshold_group.setLayout(threshold_layout)
        self.rightLayout.addWidget(self.threshold_group)
        
        # Max speakers control
        self.max_speakers_group = QGroupBox("Maximum Number of Speakers")
        max_speakers_layout = QVBoxLayout()
        
        self.max_speakers_label = QLabel(f"Max speakers: {self.max_speakers}")
        max_speakers_layout.addWidget(self.max_speakers_label)
        
        self.max_speakers_spinbox = QSpinBox()
        self.max_speakers_spinbox.setMinimum(2)
        self.max_speakers_spinbox.setMaximum(ABSOLUTE_MAX_SPEAKERS)
        self.max_speakers_spinbox.setValue(self.max_speakers)
        self.max_speakers_spinbox.valueChanged.connect(self.update_max_speakers)
        max_speakers_layout.addWidget(self.max_speakers_spinbox)
        
        self.max_speakers_explanation = QLabel(
            f"You can set between 2 and {ABSOLUTE_MAX_SPEAKERS} speakers.\n"
            "Changes will apply immediately."
        )
        self.max_speakers_explanation.setWordWrap(True)
        max_speakers_layout.addWidget(self.max_speakers_explanation)
        
        self.max_speakers_group.setLayout(max_speakers_layout)
        self.rightLayout.addWidget(self.max_speakers_group)
        
        # Speaker color legend - dynamic based on max speakers
        self.legend_group = QGroupBox("Speaker Colors")
        self.legend_layout = QVBoxLayout()
        
        # Create speaker labels dynamically
        self.speaker_labels = []
        for i in range(ABSOLUTE_MAX_SPEAKERS):
            color = SPEAKER_COLORS[i]
            color_name = SPEAKER_COLOR_NAMES[i]
            label = QLabel(f"Speaker {i+1} ({color_name}): <span style='color:{color};'>■■■■■</span>")
            self.speaker_labels.append(label)
            if i < self.max_speakers:
                self.legend_layout.addWidget(label)
        
        self.legend_group.setLayout(self.legend_layout)
        self.rightLayout.addWidget(self.legend_group)
        
        # Status display area
        self.status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Status information will be displayed here.")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        self.status_group.setLayout(status_layout)
        self.rightLayout.addWidget(self.status_group)

        # Clear button
        self.clear_button = QPushButton("Clear Conversation")
        self.clear_button.clicked.connect(self.clear_state)
        self.clear_button.setEnabled(False)
        self.rightLayout.addWidget(self.clear_button)
    
    def update_threshold(self, value):
        """Update speaker change detection threshold"""
        threshold = value / 100.0
        self.change_threshold = threshold
        self.threshold_label.setText(f"Change threshold: {threshold:.2f}")
        
        # Update in worker if it exists
        if hasattr(self, 'worker_thread'):
            self.worker_thread.set_change_threshold(threshold)
            
    def update_max_speakers(self, value):
        """Update maximum number of speakers"""
        self.max_speakers = value
        self.max_speakers_label.setText(f"Max speakers: {value}")
        
        # Update visible speaker labels
        self.update_speaker_labels()
        
        # Update in worker if it exists
        if hasattr(self, 'worker_thread'):
            self.worker_thread.set_max_speakers(value)
    
    def update_speaker_labels(self):
        """Update which speaker labels are visible based on max_speakers"""
        # Clear all labels first
        for i in range(len(self.speaker_labels)):
            label = self.speaker_labels[i]
            if label.parent():
                self.legend_layout.removeWidget(label)
                label.setParent(None)
        
        # Add only the labels for the current max_speakers
        for i in range(min(self.max_speakers, len(self.speaker_labels))):
            self.legend_layout.addWidget(self.speaker_labels[i])

    def clear_state(self):
        # Clear text edit area
        self.text_edit.clear()

        # Reset state variables
        self.displayed_text = ""
        self.last_realtime_text = ""
        self.full_sentences = []
        self.sentence_speakers = []
        self.pending_sentences = []
        
        if hasattr(self, 'worker_thread'):
            self.worker_thread.full_sentences = []
            self.worker_thread.sentence_speakers = []
            # Reset speaker detector with current threshold and max_speakers
            self.worker_thread.speaker_detector = SpeakerChangeDetector(
                embedding_dim=self.encoder.embedding_dim,
                change_threshold=self.change_threshold,
                max_speakers=self.max_speakers
            )

        # Display message
        self.text_edit.setHtml("<i>All content cleared. Waiting for new input...</i>")
    
    def update_status(self, status_text):
        self.status_label.setText(status_text)

    def showEvent(self, event):
        super().showEvent(event)
        if event.type() == QEvent.Type.Show:
            if not self.initialized:
                self.initialized = True
                self.resize(1200, 800)
                self.update_text("<i>Initializing application...</i>")

                QTimer.singleShot(500, self.init)

    def process_live_text(self, text):
        text = text.strip()

        if text:
            sentence_delimiters = '.?!。'
            prob_sentence_end = (
                len(self.last_realtime_text) > 0
                and text[-1] in sentence_delimiters
                and self.last_realtime_text[-1] in sentence_delimiters
            )

            self.last_realtime_text = text

            if prob_sentence_end:
                if FAST_SENTENCE_END:
                    self.text_retrieval_thread.recorder.stop()
                else:
                    self.text_retrieval_thread.recorder.post_speech_silence_duration = SILENCE_THRESHS[0]
            else:
                self.text_retrieval_thread.recorder.post_speech_silence_duration = SILENCE_THRESHS[1]

        self.text_detected(text)

    def text_detected(self, text):
        try:
            sentences_with_style = []
            for i, sentence in enumerate(self.full_sentences):
                sentence_text, _ = sentence
                if i >= len(self.sentence_speakers):
                    color = "#FFFFFF"  # Default white
                else:
                    speaker_id = self.sentence_speakers[i]
                    color = self.worker_thread.speaker_detector.get_color_for_speaker(speaker_id)

                sentences_with_style.append(
                    f'<span style="color:{color};">{sentence_text}</span>')

            for pending_sentence in self.pending_sentences:
                sentences_with_style.append(
                    f'<span style="color:#60FFFF;">{pending_sentence}</span>')

            new_text = " ".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text

            if new_text != self.displayed_text:
                self.displayed_text = new_text
                self.update_text(new_text)
        except Exception as e:
            print(f"Error: {e}")

    def process_final(self, text, bytes):
        text = text.strip()
        if text:
            try:
                self.pending_sentences.append(text)
                self.queue.put((text, bytes))
            except Exception as e:
                print(f"Error: {e}")

    def capture_output_and_feed_to_recorder(self):
        # Use default device settings
        device_id = str(sc.default_speaker().name)
        include_loopback = True
        
        self.recording_thread = RecordingThread(self.text_retrieval_thread.recorder)
        # Update with current device settings
        self.recording_thread.updateDevice(device_id, include_loopback)
        self.recording_thread.start()

    def recorder_ready(self):
        self.update_text("<i>Recording ready</i>")
        self.capture_output_and_feed_to_recorder()

    def init(self):
        self.update_text("<i>Loading ECAPA-TDNN model... Please wait.</i>")
        
        # Start model loading in background thread
        self.start_encoder()

    def update_loading_status(self, message):
        self.update_text(f"<i>{message}</i>")

    def start_encoder(self):
        # Create and start encoder loader thread
        self.encoder_loader_thread = EncoderLoaderThread()
        self.encoder_loader_thread.model_loaded.connect(self.on_model_loaded)
        self.encoder_loader_thread.progress_update.connect(self.update_loading_status)
        self.encoder_loader_thread.start()

    def on_model_loaded(self, encoder):
        # Store loaded encoder model
        self.encoder = encoder
        
        if self.encoder is None:
            self.update_text("<i>Failed to load ECAPA-TDNN model. Please check your configuration.</i>")
            return
        
        # Enable all controls after model is loaded
        self.clear_button.setEnabled(True)
        self.threshold_slider.setEnabled(True)
        
        # Continue initialization
        self.update_text("<i>ECAPA-TDNN model loaded. Starting recorder...</i>")
        
        self.text_retrieval_thread = TextRetrievalThread()
        self.text_retrieval_thread.recorderStarted.connect(
            self.recorder_ready)
        self.text_retrieval_thread.textRetrievedLive.connect(
            self.process_live_text)
        self.text_retrieval_thread.textRetrievedFinal.connect(
            self.process_final)
        self.text_retrieval_thread.start()
        
        self.worker_thread = SentenceWorker(
            self.queue, 
            self.encoder, 
            change_threshold=self.change_threshold,
            max_speakers=self.max_speakers
        )
        self.worker_thread.sentence_update_signal.connect(
            self.sentence_updated)
        self.worker_thread.status_signal.connect(
            self.update_status)
        self.worker_thread.start()

    def sentence_updated(self, full_sentences, sentence_speakers):
        self.pending_text = ""
        self.full_sentences = full_sentences
        self.sentence_speakers = sentence_speakers
        for sentence in self.full_sentences:
            sentence_text, _ = sentence
            if sentence_text in self.pending_sentences:
                self.pending_sentences.remove(sentence_text)
        self.text_detected("")

    def set_text(self, text):
        self.update_thread = TextUpdateThread(text)
        self.update_thread.text_update_signal.connect(self.update_text)
        self.update_thread.start()

    def update_text(self, text):
        self.text_edit.setHtml(text)
        self.text_edit.verticalScrollBar().setValue(
            self.text_edit.verticalScrollBar().maximum())


def main():
    app = QApplication(sys.argv)

    dark_stylesheet = """
    QMainWindow {
        background-color: #323232;
    }
    QTextEdit {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    """
    app.setStyleSheet(dark_stylesheet)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
