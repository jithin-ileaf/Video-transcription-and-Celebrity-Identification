import os
import pickle
import numpy as np
import cv2
import ffmpeg
import imageio_ffmpeg as iio_ffmpeg
import streamlit as st
from deepface import DeepFace
from dotenv import load_dotenv
import google.generativeai as genai

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
api_key = st.secrets["GOOGLE_API_KEY_ILEAF"]
genai.configure(api_key=api_key)

# -----------------------------
# Constants
# -----------------------------
EMBEDDINGS_FILE = "celebrity_embeddings_artists.pkl"
FRAME_SKIP = 2  # Process every 2nd frame
ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()
# -----------------------------
# Helper Functions
# -----------------------------
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def identify_celebrity(face_image, embeddings_file=EMBEDDINGS_FILE):
    """Identify celebrity from a face image (numpy array)"""
    with open(embeddings_file, "rb") as f:
        data = pickle.load(f)

    target_emb = DeepFace.represent(
        img_path=face_image,
        model_name="ArcFace",
        enforce_detection=False
    )[0]["embedding"]

    best_match = None
    best_score = 0.3  # minimum similarity threshold

    for entry in data:
        score = cosine_similarity(target_emb, entry["embedding"])
        if score > best_score:
            best_score = score
            best_match = entry["celeb"]

    return best_match, best_score


# -----------------------------
# GEMINI TRANSCRIPTION FUNCTION
# -----------------------------
def transcribe_audio_gemini(audio_path, model="models/gemini-2.5-flash"):
    """Transcribe audio using Gemini 1.5"""

    model_obj = genai.GenerativeModel(model)

    # Read audio file bytes
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Gemini accepts: FLAC, WAV, AAC, MP3, MP4, etc.
    response = model_obj.generate_content(
        contents=[
            {
                "mime_type": "audio/mp3",
                "data": audio_bytes
            },
            "Transcribe this audio exactly as spoken."
        ]
    )

    return response.text



# -----------------------------
# Video Processing
# -----------------------------
def process_video(video_path):
    """Detect celebrity in video by sampling frames"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    recognized = None

    # Calculate FPS and frames per minute
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_minute = 60 * fps
    analyzed_per_minute = frames_per_minute / FRAME_SKIP

    if 'video_stats' not in st.session_state:
        st.session_state.video_stats = {
            'fps': fps,
            'frames_per_minute': frames_per_minute,
            'analyzed_per_minute': analyzed_per_minute
        }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        try:
            match, score = identify_celebrity(frame)
            if match:
                recognized = match
                break  # Stop after first confident detection
        except Exception:
            continue

    cap.release()
    return recognized


# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Video Processing: Celebrity Recognition + Transcription (Gemini)")
st.write("Upload a video to identify celebrity faces or transcribe audio.")

# Initialize session state
if 'celebrity' not in st.session_state:
    st.session_state.celebrity = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'video_uploaded' not in st.session_state:
    st.session_state.video_uploaded = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.session_state.video_uploaded = True
    st.session_state.video_path = video_path
    st.success("Video uploaded successfully!")


# -----------------------------
# Celebrity Recognition Button
# -----------------------------
if st.session_state.video_uploaded and st.button("Identify Celebrity in Video"):
    st.info("Detecting celebrity in the video...")
    try:
        celebrity = process_video(st.session_state.video_path)
        if celebrity:
            st.session_state.celebrity = celebrity
            celebrity = celebrity.replace("_"," ")
            st.success(f"Celebrity detected: {celebrity}")
        else:
            st.warning("No celebrity face detected.")
    except Exception as e:
        st.error(f"Celebrity detection failed: {e}")

# Display celebrity result
if st.session_state.celebrity:
    st.subheader("Celebrity Identification Result:")
    st.write(st.session_state.celebrity)


# -----------------------------
# Audio Transcription Button (Gemini)
# -----------------------------
if st.session_state.video_uploaded and st.button("Transcribe Video"):
    try:
        output_audio = "temp_audio.mp3"
        ffmpeg.input(st.session_state.video_path).output(output_file).run(cmd=ffmpeg_path, overwrite_output=True)

        transcription = transcribe_audio_gemini(output_audio)
        st.session_state.transcription = transcription

        st.success("Transcription complete using Gemini!")
    except Exception as e:
        st.error(f"Transcription failed: {e}")


# Display transcription result
if st.session_state.transcription:
    st.subheader("Transcribed Text:")
    st.write(st.session_state.transcription)






