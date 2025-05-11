from flask import Flask, render_template, request, jsonify, Response, send_file, send_from_directory
import os
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import json
import traceback
import shutil
from datetime import datetime
from moviepy import VideoFileClip, AudioFileClip
import gc
import librosa

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# Ensure upload and result directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_face_landmarks(video_path, progress_callback=None):
    """Extract face landmarks with consistent frame sampling."""
    cap = None
    try:
        # Get video info using OpenCV
        duration, fps, frame_count, width, height = get_video_info(video_path)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        landmarks_list = []
        processed_frames = 0
        events = []
        
        # Define expression indices outside the processing loop so they're available everywhere
        expression_indices = [
            # Lips
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
            # Left eye
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            # Right eye
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            # Eyebrows
            70, 63, 105, 66, 107, 336, 296, 334, 293, 300
        ]
        
        # Process every frame for accuracy
        sample_rate = 1
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if progress_callback and frame_count % 10 == 0:  # Update progress every 10 frames
                event = progress_callback(frame_count, total_frames)
                if event:
                    events.append(event)
            
            if frame_count % int(sample_rate) != 0:
                continue
                
            # Resize frame for faster processing while maintaining aspect ratio
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                frame = cv2.resize(frame, (640, int(height * scale)))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                # Get the first face only
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract only the relevant landmarks
                expression_landmarks = []
                for idx in expression_indices:
                    landmark = face_landmarks.landmark[idx]
                    # Use x and y coordinates for expression comparison
                    expression_landmarks.extend([landmark.x, landmark.y])
                
                landmarks_list.append(expression_landmarks)
            else:
                # If no face detected, add empty landmarks to maintain frame alignment
                # Add a frame of zeros with the same dimensions as valid landmarks
                if landmarks_list:
                    empty_landmarks = [0.0] * len(landmarks_list[0])
                    landmarks_list.append(empty_landmarks)
                else:
                    # For the first frame, assume a standard size (number of indices * 2 coordinates)
                    empty_landmarks = [0.0] * (len(expression_indices) * 2)
                    landmarks_list.append(empty_landmarks)
            
            processed_frames += 1
        
        if not landmarks_list:
            raise ValueError("No face landmarks detected in the video")
            
        return np.array(landmarks_list), events
    except Exception as e:
        raise ValueError(f"Error processing video: {str(e)}")
    finally:
        if cap is not None:
            cap.release()

def calculate_face_expression_similarity(ref_landmarks, user_landmarks):
    """Calculate facial expression similarity score."""
    try:
        # Ensure both sequences have the same number of frames
        min_frames = min(len(ref_landmarks), len(user_landmarks))
        if min_frames == 0:
            raise ValueError("No valid frames to compare")
            
        ref_landmarks = ref_landmarks[:min_frames]
        user_landmarks = user_landmarks[:min_frames]
        
        # Calculate frame-by-frame similarity
        frame_scores = []
        for ref_frame, user_frame in zip(ref_landmarks, user_landmarks):
            # Skip frames with no face detected (all zeros)
            if np.all(ref_frame == 0) or np.all(user_frame == 0):
                frame_scores.append(0.0)
                continue
                
            # Normalize landmarks to make them invariant to face position/size
            ref_normalized = normalize_face_landmarks(ref_frame)
            user_normalized = normalize_face_landmarks(user_frame)
            
            # Calculate distance between normalized landmarks
            # Lower distance means more similar expressions
            distance = np.mean(np.sqrt(np.sum((ref_normalized - user_normalized) ** 2, axis=1)))
            
            # Convert distance to similarity score (0-100)
            # The max_expected_distance is a hyperparameter you can tune
            max_expected_distance = 0.2  # Adjust based on testing
            similarity = max(0, 100 * (1 - distance / max_expected_distance))
            
            # Ensure the score is between 0 and 100
            score = max(0, min(100, similarity))
            frame_scores.append(score)
        
        if not frame_scores:
            raise ValueError("Could not calculate facial expression similarity")
            
        # Calculate final score
        overall_score = np.mean(frame_scores)
        return round(overall_score, 2), frame_scores
    except Exception as e:
        raise ValueError(f"Error calculating face expression similarity: {str(e)}")

def normalize_face_landmarks(landmarks):
    """Normalize face landmarks to make them invariant to face position and size."""
    # Reshape to pairs of [x, y] coordinates
    landmarks_array = np.array(landmarks).reshape(-1, 2)
    
    # Center the landmarks by subtracting the mean position
    centered = landmarks_array - np.mean(landmarks_array, axis=0)
    
    # Scale to unit size
    scale = np.max(np.abs(centered))
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
        
    return normalized

def draw_face_mesh(frame, landmarks, color=(0, 255, 0), thickness=1):
    """Draw face mesh landmarks on the frame."""
    h, w = frame.shape[:2]
    
    # Reshape landmarks to [x, y] format
    reshaped_landmarks = np.array(landmarks).reshape(-1, 2)
    
    # Draw key points
    for i in range(len(reshaped_landmarks)):
        x, y = int(reshaped_landmarks[i][0] * w), int(reshaped_landmarks[i][1] * h)
        
        # Check if point is valid
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 2, color, -1)
    
    # Define connections for the main facial features (simplified)
    # You can customize this based on the MediaPipe face mesh connections
    # Here's a simplified version for eyes, eyebrows, and lips
    
    # Draw a simple outline around the eyes, eyebrows, and lips
    # This is a simplified version - you can expand it for a more detailed mesh
    
    return frame

def generate_progress_event(stage, current, total, message=""):
    """Generate SSE event for progress updates."""
    percentage = round((current / total * 100) if total > 0 else 0, 1)
    data = {
        'stage': stage,
        'current': current,
        'total': total,
        'percentage': percentage,
        'message': message
    }
    return f"data: {json.dumps(data)}\n\n"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_info(video_path):
    """Get video duration and frame count using OpenCV only."""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        return duration, fps, frame_count, width, height
    except Exception as e:
        raise ValueError(f"Error reading video: {str(e)}")
    finally:
        if cap is not None:
            cap.release()

def extract_pose_landmarks(video_path, progress_callback=None):
    """Extract pose landmarks with consistent frame sampling."""
    cap = None
    try:
        # Get video info using OpenCV
        duration, fps, frame_count, width, height = get_video_info(video_path)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        landmarks_list = []
        processed_frames = 0
        events = []
        
        # Process every frame for accuracy
        sample_rate = 1
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if progress_callback and frame_count % 10 == 0:  # Update progress every 10 frames
                event = progress_callback(frame_count, total_frames)
                if event:
                    events.append(event)
            
            if frame_count % int(sample_rate) != 0:
                continue
                
            # Resize frame for faster processing while maintaining aspect ratio
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                frame = cv2.resize(frame, (640, int(height * scale)))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Only use x and y coordinates for better comparison
                    landmarks.extend([landmark.x, landmark.y])
                landmarks_list.append(landmarks)
            
            processed_frames += 1
        
        if not landmarks_list:
            raise ValueError("No pose landmarks detected in the video")
            
        return np.array(landmarks_list), events
    except Exception as e:
        raise ValueError(f"Error processing video: {str(e)}")
    finally:
        if cap is not None:
            cap.release()

def calculate_sync_score(ref_landmarks, user_landmarks):
    """Calculate sync score with improved accuracy."""
    try:
        # Ensure both sequences have the same number of frames
        min_frames = min(len(ref_landmarks), len(user_landmarks))
        if min_frames == 0:
            raise ValueError("No valid frames to compare")
            
        ref_landmarks = ref_landmarks[:min_frames]
        user_landmarks = user_landmarks[:min_frames]
        
        # Normalize entire sequences first
        ref_normalized = (ref_landmarks - np.mean(ref_landmarks, axis=0)) / (np.std(ref_landmarks, axis=0) + 1e-6)
        user_normalized = (user_landmarks - np.mean(user_landmarks, axis=0)) / (np.std(user_landmarks, axis=0) + 1e-6)
        
        # Calculate frame-by-frame correlation
        frame_scores = []
        for ref_frame, user_frame in zip(ref_normalized, user_normalized):
            # Calculate correlation between normalized frames
            corr = np.corrcoef(ref_frame.flatten(), user_frame.flatten())[0, 1]
            
            if np.isnan(corr):
                score = 0.0
            else:
                # Calculate score (scale from -1,1 to 0,100)
                score = max(0, min(100, (corr + 1) * 50))
                
            frame_scores.append(score)
        
        if not frame_scores:
            raise ValueError("Could not calculate correlation between videos")
            
        # Calculate final score
        overall_score = np.mean(frame_scores)
        return round(overall_score, 2), frame_scores
    except Exception as e:
        raise ValueError(f"Error calculating sync score: {str(e)}")
    
def calculate_frame_sync_score(ref_landmarks, user_landmarks):
    """Calculate sync score for a single frame."""
    try:
        if not np.any(ref_landmarks) or not np.any(user_landmarks):
            return 0.0  # Return 0 if either landmark set is empty
            
        # Reshape landmarks to match expected format (33 landmarks with x,y coordinates)
        ref_reshaped = np.array(ref_landmarks).reshape(-1, 2)
        user_reshaped = np.array(user_landmarks).reshape(-1, 2)
        
        # Normalize the landmarks
        ref_normalized = (ref_reshaped - np.mean(ref_reshaped, axis=0)) / (np.std(ref_reshaped, axis=0) + 1e-6)
        user_normalized = (user_reshaped - np.mean(user_reshaped, axis=0)) / (np.std(user_reshaped, axis=0) + 1e-6)
        
        # Calculate correlation
        corr = np.corrcoef(ref_normalized.flatten(), user_normalized.flatten())[0, 1]
        
        if np.isnan(corr):
            return 0.0
            
        # Calculate score (scale from -1,1 to 0,100)
        score = max(0, min(100, (corr + 1) * 50))
        return score
    except Exception as e:
        print(f"Error calculating frame sync score: {str(e)}")
        return 0.0

def add_audio_to_video(video_path, audio_source_path, output_path):
    """
    Add audio from the audio_source_path to the video at video_path and save the result to output_path.
    Uses MoviePy instead of FFmpeg directly.
    """
    video_clip = None
    audio_source_clip = None
    audio_clip = None
    final_clip = None
    
    try:
        # Load the video clip
        video_clip = VideoFileClip(video_path)
        
        # Load the audio clip
        audio_source_clip = VideoFileClip(audio_source_path)
        audio_clip = audio_source_clip.audio
        
        # Set the audio of the video clip to the new audio clip
        final_clip = video_clip.with_audio(audio_clip)
        
        # Write the result to the output path
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        return True
    except Exception as e:
        app.logger.error(f"Error adding audio: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        # If adding audio fails, just use the video without audio
        try:
            if video_clip is not None and output_path is not None:
                # Try to write the video without audio
                video_clip.write_videofile(output_path, codec='libx264', audio=False)
        except Exception as copy_error:
            app.logger.error(f"Error copying video: {str(copy_error)}")
        
        return False
    finally:
        # Close all clips to release resources
        if video_clip is not None:
            video_clip.close()
        if audio_source_clip is not None:
            audio_source_clip.close()
        if audio_clip is not None:
            try:
                audio_clip.close()
            except:
                pass
        if final_clip is not None:
            final_clip.close()
        
        # Force garbage collection to release file handles
        gc.collect()

def safe_remove(file_path):
    """Safely remove a file with multiple attempts and proper error handling."""
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            app.logger.warning(f"Attempt {attempt+1}/{max_attempts} to remove {file_path} failed: {str(e)}")
            # Wait a bit before trying again
            time.sleep(1)
    
    app.logger.error(f"Failed to remove {file_path} after {max_attempts} attempts")
    return False

def create_comparison_video_with_skeletons(ref_video_path, user_video_path, ref_landmarks, user_landmarks, 
                                        frame_scores, output_path, pose_score, face_score, overall_score):
    """Create a side-by-side comparison video with skeletons and sync score."""
    temp_output = output_path + ".temp.mp4"
    
    try:
        # Open video captures
        ref_cap = cv2.VideoCapture(ref_video_path)
        user_cap = cv2.VideoCapture(user_video_path)
        
        # Get video properties
        ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        user_width = int(user_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        user_height = int(user_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = ref_cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate output dimensions (side by side)
        output_width = ref_width + user_width
        output_height = max(ref_height, user_height)
        
        # Create adjusted score overlay with all scores
        adjusted_pose_score = adjust_sync_score(pose_score)
        adjusted_face_score = adjust_sync_score(face_score)
        adjusted_overall_score = adjust_sync_score(overall_score)
        score_overlay = create_cool_font_score_image(
            adjusted_pose_score, 
            adjusted_face_score, 
            adjusted_overall_score, 
            output_width, 
            output_height
        )
        
        # Set up video writer - set MP4V codec (widely supported)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (output_width, output_height))
        
        # Process frames
        frame_idx = 0
        while True:
            ret1, ref_frame = ref_cap.read()
            ret2, user_frame = user_cap.read()
            
            if not ret1 or not ret2 or frame_idx >= len(frame_scores):
                break
                
            # Get the current frame's sync score and color
            current_score = frame_scores[frame_idx]
            user_skeleton_color = get_sync_color(current_score)
            ref_skeleton_color = (0, 165, 255)  # Orange for reference
            
            # Draw skeletons if landmarks are available for this frame
            if frame_idx < len(ref_landmarks) and np.any(ref_landmarks[frame_idx]):
                ref_frame = draw_skeleton(ref_frame, ref_landmarks[frame_idx], ref_skeleton_color)
            
            if frame_idx < len(user_landmarks) and np.any(user_landmarks[frame_idx]):
                user_frame = draw_skeleton(user_frame, user_landmarks[frame_idx], user_skeleton_color)
            
            # Resize frames to have the same height
            ref_frame = cv2.resize(ref_frame, (int(ref_width * output_height / ref_height), output_height))
            user_frame = cv2.resize(user_frame, (int(user_width * output_height / user_height), output_height))
            
            # Create the side-by-side frame
            combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            combined_frame[:, :ref_frame.shape[1]] = ref_frame
            combined_frame[:, ref_frame.shape[1]:ref_frame.shape[1]+user_frame.shape[1]] = user_frame
            
            # Add the score overlay
            overlay_bgr = cv2.cvtColor(score_overlay, cv2.COLOR_BGRA2BGR)
            _, _, _, alpha = cv2.split(score_overlay)
            alpha_bgr = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) / 255.0
            combined_frame = (combined_frame * (1 - alpha_bgr) + overlay_bgr * alpha_bgr).astype(np.uint8)
            
            '''
            # Add face expression similarity text above the current frame sync score
            face_text = f"Face Expression: {adjusted_face_score:.1f}%"
            cv2.putText(combined_frame, face_text, 
                      (10, output_height - 50),  # Position it above the current frame sync score
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Yellow color
            
            # Add current frame sync score text at the bottom
            current_score_text = f"Current Frame Sync: {current_score:.1f}%"
            cv2.putText(combined_frame, current_score_text, 
                      (10, output_height - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, user_skeleton_color, 2)
            '''
            # Write the frame
            out.write(combined_frame)
            frame_idx += 1
            
        # Release resources
        ref_cap.release()
        user_cap.release()
        out.release()
        
        # Let's wait a moment to make sure files are fully written and released
        time.sleep(0.5)
        
        # Try to add audio from user video using MoviePy
        add_audio_to_video(temp_output, user_video_path, output_path)
        
        # Wait a moment for resources to be released
        time.sleep(0.5)
        
        # Clean up temp file
        safe_remove(temp_output)
        
        return True
    except Exception as e:
        app.logger.error(f"Error creating comparison video: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        # In case of failure, ensure we have some output file
        try:
            if os.path.exists(temp_output) and not os.path.exists(output_path):
                # Just rename instead of trying to process with MoviePy
                os.rename(temp_output, output_path)
            elif not os.path.exists(output_path):
                # If nothing worked, copy the user video as a fallback
                shutil.copy(user_video_path, output_path)
        except Exception as fallback_error:
            app.logger.error(f"Error in fallback: {str(fallback_error)}")
        
        return False

def create_overlay_video(user_video_path, user_landmarks, frame_scores, output_path, pose_score, audio_score, overall_score):
    """Create a video with just the user's recreation and the score overlay."""
    temp_output = output_path + ".temp.mp4"
    
    try:
        # Open video capture
        user_cap = cv2.VideoCapture(user_video_path)
        
        # Get video properties
        width = int(user_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(user_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = user_cap.get(cv2.CAP_PROP_FPS)

        # Create adjusted score overlays
        adjusted_pose_score = adjust_sync_score(pose_score)
        adjusted_audio_score = adjust_sync_score(audio_score)
        adjusted_overall_score = adjust_sync_score(overall_score)
        score_overlay = create_cool_font_score_image(
            adjusted_pose_score, 
            adjusted_audio_score, 
            adjusted_overall_score, 
            width, 
            height
        )
        
        # Set up video writer - set MP4V codec (widely supported)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = user_cap.read()
            
            if not ret or frame_idx >= len(frame_scores):
                break
            
            # Get the current frame's sync score and color
            current_score = frame_scores[frame_idx]
            skeleton_color = get_sync_color(current_score)
            
            # Add the score overlay
            overlay_bgr = cv2.cvtColor(score_overlay, cv2.COLOR_BGRA2BGR)
            _, _, _, alpha = cv2.split(score_overlay)
            alpha_bgr = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) / 255.0
            frame = (frame * (1 - alpha_bgr) + overlay_bgr * alpha_bgr).astype(np.uint8)
            
            # Add current frame sync score text at the bottom
            '''
            current_score_text = f"Current Frame Sync: {current_score:.1f}%"
            cv2.putText(frame, current_score_text, 
                      (10, height - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, skeleton_color, 2)
            '''
            # Write the frame
            out.write(frame)
            frame_idx += 1
            
        # Release resources
        user_cap.release()
        out.release()
        
        # Let's wait a moment to make sure files are fully written and released
        time.sleep(0.5)
        
        # Try to add audio from user video using MoviePy
        add_audio_to_video(temp_output, user_video_path, output_path)
        
        # Wait a moment for resources to be released
        time.sleep(0.5)
        
        # Clean up temp file
        safe_remove(temp_output)
        
        return True
    except Exception as e:
        app.logger.error(f"Error creating overlay video: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        # In case of failure, ensure we have some output file
        try:
            if os.path.exists(temp_output) and not os.path.exists(output_path):
                # Just rename instead of trying to process with MoviePy
                os.rename(temp_output, output_path)
            elif not os.path.exists(output_path):
                # If nothing worked, copy the user video as a fallback
                shutil.copy(user_video_path, output_path)
        except Exception as fallback_error:
            app.logger.error(f"Error in fallback: {str(fallback_error)}")
        
        return False

def adjust_sync_score(original_score):
    """
    Adjust the sync score using the formula:
    new_score = original_score + (100 - original_score) / 2
    """
    difference = 100 - original_score
    adjustment = difference / 2
    adjusted_score = original_score + adjustment
    return round(adjusted_score, 2)  # Return the adjusted score

def get_sync_color(score):
    """Get color based on the sync score."""
    if score >= 80:  # High threshold
        return (0, 255, 0)  # Green
    elif score >= 60:  # Medium threshold
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red

def draw_skeleton(frame, landmarks, color=(0, 255, 0), thickness=2):
    """Draw a skeleton on the frame using the provided landmarks."""
    # Define connections between key body landmarks
    connections = [
        # Torso
        (11, 12),  # Shoulders
        (11, 23), (12, 24),  # Shoulders to hips
        (23, 24),  # Hips
        
        # Arms
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        
        # Legs
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28)   # Right leg
    ]
    
    h, w = frame.shape[:2]
    
    # Reshape landmarks to (33, 2) format
    reshaped_landmarks = np.array(landmarks).reshape(-1, 2)
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        
        if start_idx < len(reshaped_landmarks) and end_idx < len(reshaped_landmarks):
            start_point = (int(reshaped_landmarks[start_idx][0] * w), 
                          int(reshaped_landmarks[start_idx][1] * h))
            end_point = (int(reshaped_landmarks[end_idx][0] * w), 
                        int(reshaped_landmarks[end_idx][1] * h))
            
            # Check if points are valid
            if all(0 <= p < 50000 for p in start_point + end_point):  # Sanity check
                cv2.line(frame, start_point, end_point, color, thickness)
    
    # Draw key joints (shoulders, elbows, wrists, hips, knees, ankles)
    key_points = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    for idx in key_points:
        if idx < len(reshaped_landmarks):
            x, y = int(reshaped_landmarks[idx][0] * w), int(reshaped_landmarks[idx][1] * h)
            
            # Check if point is valid
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 5, color, -1)
    
    return frame

def extract_audio_features(video_path, progress_callback=None):
    """Extract audio features from a video file with improved feature extraction."""
    temp_audio_path = None
    try:
        # Create a temporary file for the extracted audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        if video.audio is None:
            return None, []  # No audio in video
            
        video.audio.write_audiofile(temp_audio_path, logger=None)
        video.close()
        
        # Load the audio file
        y, sr = librosa.load(temp_audio_path, sr=None)
        
        # Extract more detailed features
        # MFCCs capture timbral characteristics - use more coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # Include delta and delta-delta features (dynamics)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Chroma features relate to the harmonic content
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Temporal features - onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_features = np.array([np.mean(onset_env), np.std(onset_env)])
        
        # Combine features
        features = np.vstack([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.mean(mfcc_delta2, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.mean(spectral_rolloff),
            onset_features
        ])
        
        # Clean up temp file
        safe_remove(temp_audio_path)
        
        return features, []
    except Exception as e:
        # Clean up temp file
        if temp_audio_path:
            try:
                safe_remove(temp_audio_path)
            except:
                pass
        app.logger.error(f"Error extracting audio features: {str(e)}")
        return None, []

    
def calculate_audio_similarity(ref_features, user_features):
    """Calculate similarity between audio features with better scaling."""
    if ref_features is None or user_features is None:
        return 0.0  # No audio in one or both videos
        
    try:
        # Ensure features have the same shape by padding if necessary
        max_len = max(ref_features.shape[0], user_features.shape[0])
        if ref_features.shape[0] < max_len:
            ref_features = np.pad(ref_features, ((0, max_len - ref_features.shape[0]), (0, 0)), 'constant')
        if user_features.shape[0] < max_len:
            user_features = np.pad(user_features, ((0, max_len - user_features.shape[0]), (0, 0)), 'constant')
        
        # Normalize features
        ref_normalized = ref_features / (np.linalg.norm(ref_features) + 1e-10)
        user_normalized = user_features / (np.linalg.norm(user_features) + 1e-10)
        
        # Calculate cosine similarity
        similarity = np.dot(ref_normalized.flatten(), user_normalized.flatten()) / (
            np.linalg.norm(ref_normalized.flatten()) * np.linalg.norm(user_normalized.flatten()) + 1e-10
        )
        
        # Better scaling to 0-100
        # This moves the baseline from 50% to 0% for random audio
        score = max(0, similarity * 100)
        
        # Apply a nonlinear transformation to spread out scores
        # This makes the scoring more discriminative
        score = 100 * (score/100)**0.5
        
        return round(score, 2)
    except Exception as e:
        app.logger.error(f"Error calculating audio similarity: {str(e)}")
        return 0.0
'''
def calculate_overall_similarity(pose_score, audio_score, pose_weight=0.7, audio_weight=0.3):
    """Calculate the overall similarity score as a weighted combination of pose and audio scores."""
    if pose_score is None:
        return audio_score or 0.0
    if audio_score is None:
        return pose_score or 0.0
        
    # Apply weights and combine scores
    overall_score = (pose_score * pose_weight) + (audio_score * audio_weight)
    return round(overall_score, 2)
'''
    
def calculate_overall_similarity(pose_score, face_score, pose_weight=0.7, face_weight=0.3):
    """Calculate the overall similarity score as a weighted combination of pose and face scores."""
    if pose_score is None:
        return face_score or 0.0
    if face_score is None:
        return pose_score or 0.0
        
    # Apply weights and combine scores
    overall_score = (pose_score * pose_weight) + (face_score * face_weight)
    return round(overall_score, 2)

def create_cool_font_score_image(pose_score, face_score, overall_score, width, height):
    """Create a transparent image with just the overall score in a smaller font."""
    # Create a transparent image (RGBA)
    score_img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Set score text - only overall score
    overall_text = f"Sync Score: {overall_score:.1f}%"
    
    # Use a cool font but smaller
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.9  # Reduced from 1.2
    thickness = 2     # Reduced from 3
    color = (255, 255, 255, 255)  # White text
    
    # Position for overall score
    position_overall = (20, 40)  # Moved higher up
    
    # Draw text shadow for better visibility
    cv2.putText(score_img, overall_text, (position_overall[0]+2, position_overall[1]+2), 
                font, font_scale, (0, 0, 0, 255), thickness+1)
    
    # Draw text
    cv2.putText(score_img, overall_text, position_overall, font, font_scale, color, thickness)
    
    return score_img

def process_videos(ref_temp_path, user_temp_path, comparison_path, overlay_path):
    events = []
    
    try:
        events.append(generate_progress_event("init", 1, 6, "Reading video information..."))
        
        # Get video info
        ref_duration, ref_fps, ref_frames, ref_width, ref_height = get_video_info(ref_temp_path)
        user_duration, user_fps, user_frames, user_width, user_height = get_video_info(user_temp_path)
        
        # Extract pose landmarks from reference video
        events.append(generate_progress_event("reference", 0, ref_frames, "Processing reference video..."))
        
        def ref_progress(current, total):
            return generate_progress_event("reference", current, total)
        
        ref_pose_landmarks, ref_pose_events = extract_pose_landmarks(ref_temp_path, ref_progress)
        events.extend(ref_pose_events)
        
        # Extract pose landmarks from user video
        events.append(generate_progress_event("recreation", 0, user_frames, "Processing your video..."))
        
        def user_progress(current, total):
            return generate_progress_event("recreation", current, total)
        
        user_pose_landmarks, user_pose_events = extract_pose_landmarks(user_temp_path, user_progress)
        events.extend(user_pose_events)
        
        # Calculate pose sync score
        events.append(generate_progress_event("comparing", 3, 6, "Calculating pose sync score..."))
        pose_score, pose_frame_scores = calculate_sync_score(ref_pose_landmarks, user_pose_landmarks)
        
        # Extract face landmarks and calculate face expression similarity
        events.append(generate_progress_event("face_processing", 4, 6, "Analyzing facial expressions..."))
        
        ref_face_landmarks, _ = extract_face_landmarks(ref_temp_path)
        user_face_landmarks, _ = extract_face_landmarks(user_temp_path)
        face_score, face_frame_scores = calculate_face_expression_similarity(ref_face_landmarks, user_face_landmarks)
        
        # Calculate overall similarity score (weighted combination)
        overall_score = calculate_overall_similarity(pose_score, face_score, pose_weight=0.7, face_weight=0.3)
        
        # Calculate adjusted scores for display
        adjusted_pose_score = adjust_sync_score(pose_score)
        adjusted_face_score = adjust_sync_score(face_score)
        adjusted_overall_score = adjust_sync_score(overall_score)
        
        # Create comparison video
        events.append(generate_progress_event("creating_video", 5, 6, "Creating comparison video..."))
        create_comparison_video_with_skeletons(
            ref_temp_path, user_temp_path, 
            ref_pose_landmarks, user_pose_landmarks,
            pose_frame_scores, comparison_path, 
            pose_score, face_score, overall_score
        )
        
        # Create overlay video
        events.append(generate_progress_event("creating_overlay", 6, 6, "Creating overlay video..."))
        create_overlay_video(
            user_temp_path, user_pose_landmarks, 
            pose_frame_scores, overlay_path,
            pose_score, face_score, overall_score
        )
        
        # Send final result with all scores
        result = {
            'complete': True,
            'pose_score': {
                'original': pose_score,
                'adjusted': adjusted_pose_score
            },
            'face_score': {  # Changed from audio_score to face_score
                'original': face_score,
                'adjusted': adjusted_face_score
            },
            'overall_score': {
                'original': overall_score,
                'adjusted': adjusted_overall_score
            },
            'score': adjusted_overall_score,  # For backward compatibility
            'details': {
                'reference_duration': round(ref_duration, 2),
                'your_duration': round(user_duration, 2),
                'frames_analyzed': min(len(ref_pose_landmarks), len(user_pose_landmarks)),
                'fps': round(ref_fps, 1)
            },
            'comparison_video': os.path.basename(comparison_path),
            'overlay_video': os.path.basename(overlay_path)
        }
        events.append(f"data: {json.dumps(result)}\n\n")
        
        return events
    except Exception as e:
        # Log the full exception for debugging
        app.logger.error(f"Error in process_videos: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        error_data = {
            'error': str(e),
            'complete': True
        }
        events.append(f"data: {json.dumps(error_data)}\n\n")
        return events

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_videos():
    # Create tempfiles outside the generator
    ref_temp_path = None
    user_temp_path = None
    comparison_path = None
    overlay_path = None
    
    try:
        if 'reference_video' not in request.files or 'user_video' not in request.files:
            return jsonify({'error': 'Both videos are required'}), 400
        
        ref_video = request.files['reference_video']
        user_video = request.files['user_video']
        
        if ref_video.filename == '' or user_video.filename == '':
            return jsonify({'error': 'No selected files'}), 400
            
        if not allowed_file(ref_video.filename) or not allowed_file(user_video.filename):
            return jsonify({'error': 'Invalid file format. Allowed formats: mp4, mov, avi, mkv'}), 400
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save uploaded files to disk using secure_filename
        ref_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"ref_{timestamp}.mp4")
        user_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"user_{timestamp}.mp4")
        comparison_filename = os.path.join(app.config['RESULT_FOLDER'], f"comparison_{timestamp}.mp4")
        overlay_filename = os.path.join(app.config['RESULT_FOLDER'], f"overlay_{timestamp}.mp4")
        
        ref_temp_path = ref_filename
        user_temp_path = user_filename
        comparison_path = comparison_filename
        overlay_path = overlay_filename
        
        ref_video.save(ref_temp_path)
        user_video.save(user_temp_path)
        
        # Verify files exist and are readable
        if not os.path.exists(ref_temp_path) or not os.path.getsize(ref_temp_path) > 0:
            return jsonify({'error': 'Failed to save reference video'}), 500
            
        if not os.path.exists(user_temp_path) or not os.path.getsize(user_temp_path) > 0:
            return jsonify({'error': 'Failed to save user video'}), 500
        
        # Verify files can be opened with OpenCV
        ref_cap = cv2.VideoCapture(ref_temp_path)
        if not ref_cap.isOpened():
            ref_cap.release()
            return jsonify({'error': 'Cannot open reference video. File may be corrupted.'}), 400
        ref_cap.release()
        
        user_cap = cv2.VideoCapture(user_temp_path)
        if not user_cap.isOpened():
            user_cap.release()
            return jsonify({'error': 'Cannot open user video. File may be corrupted.'}), 400
        user_cap.release()
        
        def generate():
            try:
                # Use the process_videos function instead of duplicating the logic here
                events = process_videos(ref_temp_path, user_temp_path, comparison_path, overlay_path)
                for event in events:
                    yield event
                
                # Force garbage collection
                gc.collect()  
                
            except Exception as e:
                # Log the full exception for debugging
                app.logger.error(f"Error in generate function: {str(e)}")
                app.logger.error(traceback.format_exc())
                
                error_data = {
                    'error': str(e),
                    'complete': True
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        # Log the full exception for debugging
        app.logger.error(f"Error in upload_videos: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/result/<filename>')
def get_result(filename):
    """Serve the result video."""
    try:
        filepath = os.path.join(app.config['RESULT_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error serving result file: {str(e)}")
        return jsonify({'error': 'Could not retrieve result video'}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    app.logger.error(traceback.format_exc())
    return jsonify({'error': 'An unexpected server error occurred'}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(debug=True)