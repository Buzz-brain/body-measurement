import cv2
import numpy as np
import math
import mediapipe as mp
from flask import jsonify, request
from bson.objectid import ObjectId
from models.db import measurements_coll, fs


# --- Measurement Logic and Helpers (moved from app.py) ---
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
pose = mp_pose.Pose(model_complexity=2)
holistic = mp_holistic.Holistic()


# --- Measurement Constants ---
KNOWN_OBJECT_WIDTH_CM = 21.0
FOCAL_LENGTH = 600
DEFAULT_HEIGHT_CM = 170.0

# Load depth estimation model
def load_depth_model():
    import torch
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model

try:
    depth_model = load_depth_model()
except Exception:
    depth_model = None

# --- small helpers ---
def cm_to_inches(cm):
    return round(cm * 0.393701, 2)

def clamp_measurement(value_cm, min_ratio, max_ratio, user_height_cm):
    """Clamp a measurement in cm to a percentage range of user height."""
    min_val = user_height_cm * min_ratio
    max_val = user_height_cm * max_ratio
    return round(max(min_val, min(value_cm, max_val)), 2)

# === HELPER FUNCTIONS ===
# --- Measurement Logic ---
def calculate_effective_scale(landmarks, image_width, image_height, known_height_cm):
    """
    Blend height-based and torso-based scale factors for more accurate cm/px conversion
    Returns (effective_scale_cm_per_pixel, pixel_full_height)
    """
    mp_pose = __import__('mediapipe').solutions.pose  # Was at the top
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Average heel Y (fallback to ankles if needed)
    heel_y = None
    try:
        heel_y = (left_heel.y + right_heel.y) / 2.0
    except Exception:
        # fallback to ankles
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        heel_y = (left_ankle.y + right_ankle.y) / 2.0

    # --- Height scale (cm per pixel) ---
    pixel_height = abs((nose.y * image_height) - (heel_y * image_height))
    height_scale = known_height_cm / pixel_height if pixel_height > 0 else 1.0

    # --- Torso scale (shoulder → hip vertical distance) ---
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    avg_hip_y = (left_hip.y + right_hip.y) / 2.0
    torso_pixels = abs((avg_hip_y - avg_shoulder_y) * image_height)

    # Anthropometric reference: torso ≈ 0.26 * body height
    expected_torso_cm = known_height_cm * 0.26
    torso_scale = expected_torso_cm / torso_pixels if torso_pixels > 0 else height_scale

    # --- Blend both scales ---
    effective_scale = (0.7 * height_scale) + (0.3 * torso_scale)
    return effective_scale, pixel_height

def validate_front_image(frame):
    """
    Validate that the front image is suitable for processing:
    - Check size
    - Check if pose landmarks are detectable
    """
    import cv2, numpy as np, mediapipe as mp # Was at the top
    holistic = mp.solutions.holistic.Holistic() # Was at the top
    if frame is None or frame.size == 0:
        return False, "Invalid image uploaded."
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    if not results.pose_landmarks:
        return False, "No pose detected in the front image. Ensure full body is visible."
    return True, ""

# === MEASUREMENTS ===
def calculate_gender_specific_measurements(landmarks, scale_factor, image_width, image_height, gender, frame=None, user_height_cm=DEFAULT_HEIGHT_CM):
    """Enhanced measurements with gender-specific formulas. estimated_height uses user input."""
    measurements = {}

    def pixel_to_cm(px):
        return round(px * scale_factor, 2)
    
    # Key landmark positions
    mp_pose = __import__('mediapipe').solutions.pose
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]

    # Height in pixels (nose->heel) and pixel-derived cm (kept for proportional calculations)
    height_pixels = abs((nose.y * image_height) - (left_heel.y * image_height))
    height_cm_pixel = pixel_to_cm(height_pixels)

    # Shoulder width
    shoulder_width_px = abs((left_shoulder.x * image_width) - (right_shoulder.x * image_width))
    shoulder_width_cm = pixel_to_cm(shoulder_width_px)

    # Clamp shoulder width by anthropometric ranges to avoid perspective outliers
    if gender == "male":
        shoulder_width_cm = clamp_measurement(shoulder_width_cm, 0.24, 0.32, user_height_cm)
    else:
        shoulder_width_cm = clamp_measurement(shoulder_width_cm, 0.23, 0.28, user_height_cm)

    # Compute gender-specific measurements (pass user height for length references)
    if gender == "male":
        measurements.update(
            calculate_male_measurements(
                landmarks, scale_factor, image_width, image_height,
                height_cm_pixel, shoulder_width_cm, frame, user_height_cm
            )
        )
    else:
        measurements.update(
            calculate_female_measurements(
                landmarks, scale_factor, image_width, image_height,
                height_cm_pixel, shoulder_width_cm, frame, user_height_cm
            )
        )

    # Use user-provided height for estimated height to avoid drift
    measurements["estimated_height"] = {
        "cm": round(user_height_cm, 1),
        "inches": cm_to_inches(user_height_cm)
    }
    measurements["shoulder_width"] = {
        "cm": shoulder_width_cm,
        "inches": cm_to_inches(shoulder_width_cm)
    }

    return measurements

def calculate_male_measurements(landmarks, scale_factor, image_width, image_height,
                                height_cm, shoulder_width_cm, frame, user_height_cm):
    """Male-specific measurements including sleeves, lower body and lengths computed from shoulder down."""
    measurements = {}

    def pixel_to_cm(px):
        return round(px * scale_factor, 2)
    
    def calculate_distance(l1, l2):
        x1, y1 = l1.x * image_width, l1.y * image_height
        x2, y2 = l2.x * image_width, l2.y * image_height
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Landmarks used
    mp_pose = __import__('mediapipe').solutions.pose # Was at the top
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    # Torso circumferences (from shoulder width)
    chest_circumference = shoulder_width_cm * 2.05
    chest_circumference = clamp_measurement(chest_circumference, 0.45, 0.85, user_height_cm)
    waist_circumference = clamp_measurement(chest_circumference * 0.85, 0.35, 0.7, user_height_cm)
    hip_circumference = clamp_measurement(chest_circumference * 1.05, 0.45, 0.9, user_height_cm)

    measurements["chest_circumference"] = {"cm": chest_circumference, "inches": cm_to_inches(chest_circumference)}
    measurements["waist_circumference"] = {"cm": waist_circumference, "inches": cm_to_inches(waist_circumference)}
    measurements["hip_circumference"] = {"cm": hip_circumference, "inches": cm_to_inches(hip_circumference)}
    
    # Sleeve measurements: distances in pixels scaled to cm
    upper_arm_px = calculate_distance(left_shoulder, left_elbow)
    full_arm_px = calculate_distance(left_shoulder, left_wrist)
    short_sleeve_cm = pixel_to_cm(upper_arm_px)
    three_quarter_sleeve_cm = pixel_to_cm(upper_arm_px * 1.5)
    long_sleeve_cm = pixel_to_cm(full_arm_px)

    # clamp sleeve lengths relative to height
    short_sleeve_cm = clamp_measurement(short_sleeve_cm, 0.12, 0.30, user_height_cm)
    three_quarter_sleeve_cm = clamp_measurement(three_quarter_sleeve_cm, 0.20, 0.45, user_height_cm)
    long_sleeve_cm = clamp_measurement(long_sleeve_cm, 0.35, 0.6, user_height_cm)

    measurements["short_sleeve_length"] = {"cm": round(short_sleeve_cm, 1), "inches": cm_to_inches(short_sleeve_cm)}
    measurements["three_quarter_sleeve"] = {"cm": round(three_quarter_sleeve_cm, 1), "inches": cm_to_inches(three_quarter_sleeve_cm)}
    measurements["long_sleeve_length"] = {"cm": round(long_sleeve_cm, 1), "inches": cm_to_inches(long_sleeve_cm)}

    # Biceps circumference estimate
    biceps_cm = clamp_measurement(pixel_to_cm(upper_arm_px) * 0.7, 0.07, 0.25, user_height_cm)
    measurements["biceps_circumference"] = {"cm": round(biceps_cm, 1), "inches": cm_to_inches(biceps_cm)}

    # Lower body
    inseam_px = calculate_distance(left_hip, left_ankle)
    thigh_px = calculate_distance(left_hip, left_knee)

    inseam_cm = pixel_to_cm(inseam_px)
    thigh_circumference_cm = pixel_to_cm(thigh_px * 0.6)
    inseam_cm = clamp_measurement(inseam_cm, 0.35, 0.55, user_height_cm)
    thigh_circumference_cm = clamp_measurement(thigh_circumference_cm, 0.12, 0.32, user_height_cm)

    measurements["inseam"] = {"cm": round(inseam_cm, 1), "inches": cm_to_inches(inseam_cm)}
    measurements["thigh_circumference"] = {"cm": round(thigh_circumference_cm, 1), "inches": cm_to_inches(thigh_circumference_cm)}

    # Lengths: TOP (shoulder -> hip), FULL (shoulder -> heel/ankle)
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    hip_center_y = (left_hip.y + right_hip.y) / 2.0
    
    # determine heel y (prefer heels, fallback to ankles)
    try:
        left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        heel_y = (left_heel.y + right_heel.y) / 2.0
    except Exception:
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        heel_y = (left_ankle.y + right_ankle.y) / 2.0

    top_length_px = abs((avg_shoulder_y - hip_center_y) * image_height)
    full_length_px = abs((avg_shoulder_y - heel_y) * image_height)
    
    top_length_cm = pixel_to_cm(top_length_px)
    full_length_cm = pixel_to_cm(full_length_px)

    # clamp lengths sensibly relative to user height
    top_length_cm = clamp_measurement(top_length_cm, 0.12, 0.6, user_height_cm)
    full_length_cm = clamp_measurement(full_length_cm, 0.4, 1.0, user_height_cm)

    measurements["top_length"] = {"cm": round(top_length_cm, 1), "inches": cm_to_inches(top_length_cm)}
    measurements["full_length"] = {"cm": round(full_length_cm, 1), "inches": cm_to_inches(full_length_cm)}
    return measurements

def calculate_female_measurements(landmarks, scale_factor, image_width, image_height,
                                  height_cm, shoulder_width_cm, frame, user_height_cm):
    """Female-specific measurements including sleeve and lower-body outputs."""
    measurements = {}

    def pixel_to_cm(px):
        return round(px * scale_factor, 2)
    
    def calculate_distance(l1, l2):
        x1, y1 = l1.x * image_width, l1.y * image_height
        x2, y2 = l2.x * image_width, l2.y * image_height
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Landmarks
    mp_pose = __import__('mediapipe').solutions.pose #Was at the top
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    
    # Torso circumferences
    bust_circumference = shoulder_width_cm * 2.03
    bust_circumference = clamp_measurement(bust_circumference, 0.45, 0.9, user_height_cm)
    waist_circumference = clamp_measurement(bust_circumference * 0.80, 0.35, 0.65, user_height_cm)
    hip_circumference = clamp_measurement(bust_circumference * 1.066, 0.45, 0.95, user_height_cm)

    measurements["bust_circumference"] = {"cm": bust_circumference, "inches": cm_to_inches(bust_circumference)}
    measurements["waist_circumference"] = {"cm": waist_circumference, "inches": cm_to_inches(waist_circumference)}
    measurements["hip_circumference"] = {"cm": hip_circumference, "inches": cm_to_inches(hip_circumference)}

    # Sleeves
    upper_arm_px = calculate_distance(left_shoulder, left_elbow)
    full_arm_px = calculate_distance(left_shoulder, left_wrist)

    short_sleeve_cm = pixel_to_cm(upper_arm_px)
    three_quarter_sleeve_cm = pixel_to_cm(upper_arm_px * 1.5)
    long_sleeve_cm = pixel_to_cm(full_arm_px)
    
    short_sleeve_cm = clamp_measurement(short_sleeve_cm, 0.12, 0.30, user_height_cm)
    three_quarter_sleeve_cm = clamp_measurement(three_quarter_sleeve_cm, 0.20, 0.45, user_height_cm)
    long_sleeve_cm = clamp_measurement(long_sleeve_cm, 0.35, 0.6, user_height_cm)

    measurements["short_sleeve_length"] = {"cm": round(short_sleeve_cm, 1), "inches": cm_to_inches(short_sleeve_cm)}
    measurements["three_quarter_sleeve"] = {"cm": round(three_quarter_sleeve_cm, 1), "inches": cm_to_inches(three_quarter_sleeve_cm)}
    measurements["long_sleeve_length"] = {"cm": round(long_sleeve_cm, 1), "inches": cm_to_inches(long_sleeve_cm)}

    # Biceps
    biceps_cm = clamp_measurement(pixel_to_cm(upper_arm_px) * 0.6, 0.07, 0.25, user_height_cm)
    measurements["biceps_circumference"] = {"cm": round(biceps_cm, 1), "inches": cm_to_inches(biceps_cm)}

    # Lower body
    left_hip_l = left_hip
    inseam_px = calculate_distance(left_hip_l, left_ankle)
    thigh_px = calculate_distance(left_hip_l, left_knee)

    inseam_cm = pixel_to_cm(inseam_px)
    thigh_circumference_cm = pixel_to_cm(thigh_px * 0.6)

    inseam_cm = clamp_measurement(inseam_cm, 0.35, 0.55, user_height_cm)
    thigh_circumference_cm = clamp_measurement(thigh_circumference_cm, 0.12, 0.32, user_height_cm)

    measurements["inseam"] = {"cm": round(inseam_cm, 1), "inches": cm_to_inches(inseam_cm)}
    measurements["thigh_circumference"] = {"cm": round(thigh_circumference_cm, 1), "inches": cm_to_inches
    (thigh_circumference_cm)}

    # Lengths from shoulder down
    avg_shoulder_y = (left_shoulder.y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2.0
    hip_center_y = (left_hip.y + right_hip.y) / 2.0

    # heel y fallback
    try:
        left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        heel_y = (left_heel.y + right_heel.y) / 2.0
    except Exception:
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        heel_y = (left_ankle.y + right_ankle.y) / 2.0

    top_length_px = abs((avg_shoulder_y - hip_center_y) * image_height)
    full_length_px = abs((avg_shoulder_y - heel_y) * image_height)

    top_length_cm = pixel_to_cm(top_length_px)
    full_length_cm = pixel_to_cm(full_length_px)

    top_length_cm = clamp_measurement(top_length_cm, 0.12, 0.6, user_height_cm)
    full_length_cm = clamp_measurement(full_length_cm, 0.4, 1.0, user_height_cm)

    measurements["top_length"] = {"cm": round(top_length_cm, 1), "inches": cm_to_inches(top_length_cm)}
    measurements["full_length"] = {"cm": round(full_length_cm, 1), "inches": cm_to_inches(full_length_cm)}
    return measurements









# Placeholder for process_images_and_get_measurements

def process_images_and_get_measurements(front_frame, side_frame, gender, user_height_cm):
    # Move the full implementation from app.py
    pass

# Example stub for get_user_measurements (replace with full logic):
def get_user_measurements(request):
    user_id = request.user['user_id']
    user_role = request.user['role']
    query = {'user_id': user_id}
    records = list(measurements_coll.find(query).sort('created_at', -1))
    for r in records:
        r['_id'] = str(r['_id'])
        if 'front_image_id' in r and r['front_image_id']:
            r['front_image_id'] = str(r['front_image_id'])
        if 'side_image_id' in r and r['side_image_id']:
            r['side_image_id'] = str(r['side_image_id'])
    return jsonify({'records': records})



# Stub for create_measurement_record
def create_measurement_record(request):
    """
    Process uploaded front and side images, validate, calculate measurements, and store results in MongoDB.
    Expects: multipart/form-data with 'front', 'side', 'gender', 'height_unit', 'height_ft', 'height_in', 'height_cm'.
    Returns: JSON with measurement results or error.
    """
    import cv2, numpy as np # Was at the top
    from models.db import measurements_coll, fs # Was at the top
    from datetime import datetime # Was at the top


    # Validate files
    if 'front' not in request.files or 'side' not in request.files:
        return jsonify({'error': 'Both front and side images are required.'}), 400
    
    front_image_file = request.files['front']
    side_image_file = request.files['side']
    gender = request.form.get('gender', 'male')

    if front_image_file.filename == '' or side_image_file.filename == '':
        return jsonify({'error': 'Please select both front and side images.'}), 400
    
    # Process height input
    height_unit = request.form.get('height_unit', 'cm')
    try:
        if height_unit == 'ft':
            height_ft = float(request.form.get('height_ft') or 0)
            height_in = float(request.form.get('height_in') or 0)
            user_height_cm = (height_ft * 30.48) + (height_in * 2.54)
        else:
            user_height_cm = float(request.form.get('height_cm') or 0)
        if user_height_cm == 0:
            user_height_cm = DEFAULT_HEIGHT_CM
    except Exception:
        user_height_cm = DEFAULT_HEIGHT_CM

    try:
        # Read images
        front_image_data = front_image_file.read()
        front_image_np = np.frombuffer(front_image_data, np.uint8)
        front_frame = cv2.imdecode(front_image_np, cv2.IMREAD_COLOR)

        side_image_data = side_image_file.read()
        side_image_np = np.frombuffer(side_image_data, np.uint8)
        side_frame = cv2.imdecode(side_image_np, cv2.IMREAD_COLOR)

        # Validate front image
        is_valid, error_msg = validate_front_image(front_frame)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # MediaPipe
        import mediapipe as mp # Was at the top
        holistic = mp.solutions.holistic.Holistic()
        front_rgb = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
        front_results = holistic.process(front_rgb)

        if not front_results.pose_landmarks:
            return jsonify({'error': 'Could not detect pose in front image.'}), 400
        
        # Calculate effective scale (cm/pixel)
        image_height, image_width = front_frame.shape[:2]
        scale_factor, pixel_height = calculate_effective_scale(
            front_results.pose_landmarks.landmark,
            image_width,
            image_height,
            user_height_cm
        )

        # Generate measurements (pass user_height_cm so lengths are computed from shoulders)
        measurements = calculate_gender_specific_measurements(
            front_results.pose_landmarks.landmark,
            scale_factor,
            image_width,
            image_height,
            gender,
            front_frame,
            user_height_cm
        )

        # Store images in GridFS
        front_image_id = fs.put(front_image_data, filename=front_image_file.filename, content_type=front_image_file.content_type)
        side_image_id = fs.put(side_image_data, filename=side_image_file.filename, content_type=side_image_file.content_type)

        # Store measurement record
        record = {
            'user_id': getattr(request, 'user', {}).get('user_id', None),
            'gender': gender,
            'height_cm': user_height_cm,
            'front_image_id': front_image_id,
            'side_image_id': side_image_id,
            'measurements': measurements,
            'created_at': datetime.utcnow()
        }

        result = measurements_coll.insert_one(record)
        record['_id'] = str(result.inserted_id)
        record['front_image_id'] = str(front_image_id)
        record['side_image_id'] = str(side_image_id)

        return jsonify({'record': record, 'measurements': measurements}), 201
    except Exception as e:
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500


# Stub for get_measurement_record (replace with full logic)
def get_measurement_record(record_id):
    # TODO: Implement logic to fetch a single measurement record
    return jsonify({'message': f'Measurement record {record_id} not yet implemented.'}), 501

# Stub for get_image (replace with full logic)
def get_image(image_id):
    # TODO: Implement logic to fetch and serve image
    return jsonify({'message': f'Image {image_id} not yet implemented.'}), 501





































# 3D AVAATAR

# import React, { useRef, useEffect, useState } from 'react';
# import { MeasurementRecord } from '../../types';
# import * as THREE from 'three';
# import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';


# interface Avatar3DProps {
#   measurements: MeasurementRecord;
#   className?: string;
#   currentView?: 'front' | 'side';
#   activeMeasurement?: string; // measurement id or type
#   onHighlightMeasurement?: (measurementId: string | null) => void;
# }

# export const Avatar3D: React.FC<Avatar3DProps> = ({ measurements, className = '', currentView = 'front', activeMeasurement, onHighlightMeasurement }) => {
#   // For highlight/tooltip
#   const [highlighted, setHighlighted] = useState<string | null>(null);
#   // Map measurement type to mesh/bone name
#   const measurementToMesh: Record<string, string[]> = {
#     chest: ['Wolf3D_Outfit_Top', 'Wolf3D_Body'],
#     chest_circumference: ['Wolf3D_Outfit_Top', 'Wolf3D_Body'],
#     waist: ['Wolf3D_Outfit_Top', 'Wolf3D_Outfit_Bottom', 'Wolf3D_Body'],
#     waist_circumference: ['Wolf3D_Outfit_Top', 'Wolf3D_Outfit_Bottom', 'Wolf3D_Body'],
#     hip: ['Wolf3D_Outfit_Bottom', 'Wolf3D_Body'],
#     hip_circumference: ['Wolf3D_Outfit_Bottom', 'Wolf3D_Body'],
#     shoulder_width: ['Wolf3D_Head', 'Wolf3D_Body'],
#     inseam: ['Wolf3D_Body'],
#     thigh_circumference: ['Wolf3D_Body', 'Wolf3D_Outfit_Bottom'],
#     biceps_circumference: ['Wolf3D_Body', 'Wolf3D_Outfit_Top'],
#     short_sleeve_length: ['Wolf3D_Body', 'Wolf3D_Outfit_Top'],
#     three_quarter_sleeve: ['Wolf3D_Body', 'Wolf3D_Outfit_Top'],
#     long_sleeve_length: ['Wolf3D_Body', 'Wolf3D_Outfit_Top'],
#     top_length: ['Wolf3D_Outfit_Top', 'Wolf3D_Body'],
#     full_length: ['Wolf3D_Body', 'Wolf3D_Outfit_Bottom'],
#     estimated_height: ['Wolf3D_Body'],
#     height: ['Wolf3D_Body'],
#     // fallback
#     default: ['Wolf3D_Body']
#   };
#   const mountRef = useRef<HTMLDivElement>(null);
#   const [rotation, setRotation] = useState({ x: 0, y: 0 });
#   const [isDragging, setIsDragging] = useState(false);
#   const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
#   const modelRef = useRef<THREE.Group | null>(null);
#   // Store bone/mesh references for morphing
#   const boneRefs = useRef<{ [key: string]: THREE.Bone | undefined }>({});
#   const meshRefs = useRef<{ [key: string]: THREE.Mesh | undefined }>({});

#   // Helper to extract measurement values
#   const findBy = (type: string): number | undefined => {
#       (gltf: any) => {
#         const model = gltf.scene;
#         // Compute bounding box to center and fit model
#         const box = new THREE.Box3().setFromObject(model);
#         const size = new THREE.Vector3();
#         const center = new THREE.Vector3();
#         box.getSize(size);
#         box.getCenter(center);
#         // Center model at origin, then offset slightly down so both head and feet are visible
#         model.position.x += (model.position.x - center.x);
#         model.position.z += (model.position.z - center.z);
#         model.position.y += (model.position.y - center.y) - (size.y / 4);
#         // Scale model to fit container height (assume minHeight: 400px)
#         const desiredHeight = 2.8; // world units, tweak as needed
#         const scale = desiredHeight / size.y;
#         model.scale.set(scale, scale, scale);

#         // Store references to key bones and meshes for morphing
#         const boneNames = [
#           'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
#           'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
#           'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
#           'LeftUpLeg', 'LeftLeg', 'LeftFoot',
#           'RightUpLeg', 'RightLeg', 'RightFoot',
#         ];
#         const meshNames = [
#           'Wolf3D_Body', 'Wolf3D_Head', 'Wolf3D_Outfit_Top', 'Wolf3D_Outfit_Bottom', 'Wolf3D_Outfit_Footwear',
#         ];
#         model.traverse((obj: any) => {
#           if (obj.isBone && boneNames.includes(obj.name)) {
#             boneRefs.current[obj.name] = obj;
#           }
#           if (obj.isMesh && meshNames.includes(obj.name)) {
#             meshRefs.current[obj.name] = obj;
#           }
#         });

#         modelRef.current = model;
#         scene.add(model);

#         // Initial morphing and highlight will be handled by top-level useEffect
#         camera.position.set(0, 0.15, size.y * scale * 1.35);
#         useEffect(() => {
#           // Defensive: only run if model and meshRefs are loaded
#           if (!modelRef.current || !meshRefs.current || Object.keys(meshRefs.current).length === 0) return;
#           // Remove previous highlight
#           Object.values(meshRefs.current).forEach(mesh => {
#             if (mesh) {
#               const mat = mesh.material;
#               if ((mat as any).isMeshStandardMaterial) {
#                 (mat as THREE.MeshStandardMaterial).emissive.setHex(0x000000);
#                 (mat as THREE.MeshStandardMaterial).emissiveIntensity = 0;
#               }
#             }
#           });
#           // Only highlight if a measurement is selected
#           let highlightType = activeMeasurement || highlighted;
#           if (highlightType) {
#             const m = measurements.measurements.find(m => m.id === highlightType || m.type === highlightType);
#             if (m) {
#               const measurementToMesh: Record<string, string[]> = {
#                 chest: ['Wolf3D_Outfit_Top', 'Wolf3D_Body'],
#                 chest_circumference: ['Wolf3D_Outfit_Top', 'Wolf3D_Body'],
#                 waist: ['Wolf3D_Outfit_Top', 'Wolf3D_Outfit_Bottom', 'Wolf3D_Body'],
#                 waist_circumference: ['Wolf3D_Outfit_Top', 'Wolf3D_Outfit_Bottom', 'Wolf3D_Body'],
#                 hip: ['Wolf3D_Outfit_Bottom', 'Wolf3D_Body'],
#                 hip_circumference: ['Wolf3D_Outfit_Bottom', 'Wolf3D_Body'],
#                 shoulder_width: ['Wolf3D_Head', 'Wolf3D_Body'],
#                 inseam: ['Wolf3D_Body'],
#                 thigh_circumference: ['Wolf3D_Body', 'Wolf3D_Outfit_Bottom'],
#                 biceps_circumference: ['Wolf3D_Body', 'Wolf3D_Outfit_Top'],
#                 short_sleeve_length: ['Wolf3D_Body', 'Wolf3D_Outfit_Top'],
#                 three_quarter_sleeve: ['Wolf3D_Body', 'Wolf3D_Outfit_Top'],
#                 long_sleeve_length: ['Wolf3D_Body', 'Wolf3D_Outfit_Top'],
#                 top_length: ['Wolf3D_Outfit_Top', 'Wolf3D_Body'],
#                 full_length: ['Wolf3D_Body', 'Wolf3D_Outfit_Bottom'],
#                 estimated_height: ['Wolf3D_Body'],
#                 height: ['Wolf3D_Body'],
#                 // fallback
#                 default: ['Wolf3D_Body']
#               };
#               const meshNames = measurementToMesh[m.type] || [];
#               const highlightPulse = 0.7 + 0.3 * Math.abs(Math.sin(Date.now() / 400));
#               meshNames.forEach(name => {
#                 const mesh = meshRefs.current[name];
#                 if (mesh) {
#                   const mat = mesh.material;
#                   if ((mat as any).isMeshStandardMaterial) {
#                     (mat as THREE.MeshStandardMaterial).emissive.setHex(0xffd700); // gold highlight
#                     (mat as THREE.MeshStandardMaterial).emissiveIntensity = highlightPulse;
#                   }
#                 }
#               });
#             }
#           }
#         }, [activeMeasurement, highlighted, measurements]);
#         model.position.z += (model.position.z - center.z);
#         model.position.y += (model.position.y - center.y) - (size.y / 4);
#         // Scale model to fit container height (assume minHeight: 400px)
#         const desiredHeight = 2.8; // world units, tweak as needed
#         const scale = desiredHeight / size.y;
#         model.scale.set(scale, scale, scale);

#         // Store references to key bones and meshes for morphing
#         const boneNames = [
#           'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
#           'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
#           'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
#           'LeftUpLeg', 'LeftLeg', 'LeftFoot',
#           'RightUpLeg', 'RightLeg', 'RightFoot',
#         ];
#         const meshNames = [
#           'Wolf3D_Body', 'Wolf3D_Head', 'Wolf3D_Outfit_Top', 'Wolf3D_Outfit_Bottom', 'Wolf3D_Outfit_Footwear',
#         ];
#         model.traverse((obj: any) => {
#           if (obj.isBone && boneNames.includes(obj.name)) {
#             boneRefs.current[obj.name] = obj;
#           }
#           if (obj.isMesh && meshNames.includes(obj.name)) {
#             meshRefs.current[obj.name] = obj;
#           }
#         });

#         modelRef.current = model;
#         scene.add(model);

#   // Initial morphing and highlight
#   applyMorphing();
#   updateHighlight();
#         camera.position.set(0, 0.15, size.y * scale * 1.35);
#         camera.lookAt(0, 0, 0);
#         modelRef.current = model;
#         scene.add(model);
#       },
#       undefined,
#       (error: any) => {
#         console.error('Error loading human GLB:', error);
#       }
#     );

#   // Renderer
#     // Morphing function: apply transforms based on measurements
#     function applyMorphing() {
#       if (!modelRef.current) return;
#       // Example: get user measurements (in cm)
#       const userHeight = findBy('height');
#       const userShoulder = findBy('shoulder_width');
#       const userHip = findBy('hip_width');
#       const userChest = findBy('chest_circumference');
#       const userWaist = findBy('waist_circumference');
#       const userArm = findBy('arm_length');
#       const userLeg = findBy('leg_length');

#       // Default avatar values (adjust to match your model's real-world scale)
#       const defaultHeight = 175; // cm
#       const defaultShoulder = 42; // cm
#       const defaultHip = 38; // cm
#       const defaultChest = 90; // cm
#       const defaultWaist = 75; // cm
#       const defaultArm = 60; // cm
#       const defaultLeg = 90; // cm

#       // Height: scale Y of root
#       if (userHeight && boneRefs.current['Hips']) {
#         const scaleY = userHeight / defaultHeight;
#         boneRefs.current['Hips'].parent?.scale.setY(scaleY);
#       }
#       // Shoulder width: move shoulders
#       if (userShoulder && boneRefs.current['LeftShoulder'] && boneRefs.current['RightShoulder']) {
#         const shoulderScale = userShoulder / defaultShoulder;
#         boneRefs.current['LeftShoulder'].position.x = Math.abs(boneRefs.current['LeftShoulder'].position.x) * shoulderScale;
#         boneRefs.current['RightShoulder'].position.x = -Math.abs(boneRefs.current['RightShoulder'].position.x) * shoulderScale;
#       }
#       // Hip width: move upper legs
#       if (userHip && boneRefs.current['LeftUpLeg'] && boneRefs.current['RightUpLeg']) {
#         const hipScale = userHip / defaultHip;
#         boneRefs.current['LeftUpLeg'].position.x = Math.abs(boneRefs.current['LeftUpLeg'].position.x) * hipScale;
#         boneRefs.current['RightUpLeg'].position.x = -Math.abs(boneRefs.current['RightUpLeg'].position.x) * hipScale;
#       }
#       // Chest/waist/hip circumference: scale X/Z of body mesh
#       if (meshRefs.current['Wolf3D_Body']) {
#         if (userChest) meshRefs.current['Wolf3D_Body'].scale.x = userChest / defaultChest;
#         if (userWaist) meshRefs.current['Wolf3D_Body'].scale.z = userWaist / defaultWaist;
#       }
#       // Arm length: scale Y of arm bones
#       if (userArm) {
#         const armScale = userArm / defaultArm;
#         ['LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand'].forEach(bone => {
#           if (boneRefs.current[bone]) boneRefs.current[bone]!.scale.y = armScale;
#         });
#       }
#       // Leg length: scale Y of leg bones
#       if (userLeg) {
#         const legScale = userLeg / defaultLeg;
#         ['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot'].forEach(bone => {
#           if (boneRefs.current[bone]) boneRefs.current[bone]!.scale.y = legScale;
#         });
#       }
#     }
#     const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
#     renderer.setSize(width, heightPx);
#     mountRef.current.appendChild(renderer.domElement);


#     // Animation loop
#     let frameId: number;
#     const animate = () => {
#       // Rotate the loaded model if present
#       if (modelRef.current) {
#         modelRef.current.rotation.y = rotation.y * 0.02;
#         modelRef.current.rotation.x = rotation.x * 0.02;
#       }
#       // Update highlight every frame (in case prop changes)
#       if (typeof updateHighlight === 'function') updateHighlight();
#       renderer.render(scene, camera);
#       frameId = requestAnimationFrame(animate);
#     };
#     animate();

#     // Handle resize
#     const handleResize = () => {
#       const w = mountRef.current?.offsetWidth || 400;
#       const h = mountRef.current?.offsetHeight || 400;
#       renderer.setSize(w, h);
#       camera.aspect = w / h;
#       camera.updateProjectionMatrix();
#     };
#     window.addEventListener('resize', handleResize);

#     return () => {
#       cancelAnimationFrame(frameId);
#       window.removeEventListener('resize', handleResize);
#       renderer.dispose();
#       mountRef.current && (mountRef.current.innerHTML = '');
#     };
#     // eslint-disable-next-line
#     // Re-apply morphing when measurements change
#     // eslint-disable-next-line
#   }, [measurements, rotation, currentView]);

#   // Re-apply morphing when measurements change (after model is loaded)
#   useEffect(() => {
#     if (modelRef.current) {
#       // Defensive: only morph if bones/meshes are available
#       if (Object.keys(boneRefs.current).length > 0) {
#         // @ts-ignore
#         const applyMorphing = () => {
#           // Same as above
#           const userHeight = findBy('height');
#           const userShoulder = findBy('shoulder_width');
#           const userHip = findBy('hip_width');
#           const userChest = findBy('chest_circumference');
#           const userWaist = findBy('waist_circumference');
#           const userArm = findBy('arm_length');
#           const userLeg = findBy('leg_length');
#           const defaultHeight = 175;
#           const defaultShoulder = 42;
#           const defaultHip = 38;
#           const defaultChest = 90;
#           const defaultWaist = 75;
#           const defaultArm = 60;
#           const defaultLeg = 90;
#           if (userHeight && boneRefs.current['Hips']) {
#             const scaleY = userHeight / defaultHeight;
#             boneRefs.current['Hips'].parent?.scale.setY(scaleY);
#           }
#           if (userShoulder && boneRefs.current['LeftShoulder'] && boneRefs.current['RightShoulder']) {
#             const shoulderScale = userShoulder / defaultShoulder;
#             boneRefs.current['LeftShoulder'].position.x = Math.abs(boneRefs.current['LeftShoulder'].position.x) * shoulderScale;
#             boneRefs.current['RightShoulder'].position.x = -Math.abs(boneRefs.current['RightShoulder'].position.x) * shoulderScale;
#           }
#           if (userHip && boneRefs.current['LeftUpLeg'] && boneRefs.current['RightUpLeg']) {
#             const hipScale = userHip / defaultHip;
#             boneRefs.current['LeftUpLeg'].position.x = Math.abs(boneRefs.current['LeftUpLeg'].position.x) * hipScale;
#             boneRefs.current['RightUpLeg'].position.x = -Math.abs(boneRefs.current['RightUpLeg'].position.x) * hipScale;
#           }
#           if (meshRefs.current['Wolf3D_Body']) {
#             if (userChest) meshRefs.current['Wolf3D_Body'].scale.x = userChest / defaultChest;
#             if (userWaist) meshRefs.current['Wolf3D_Body'].scale.z = userWaist / defaultWaist;
#           }
#           if (userArm) {
#             const armScale = userArm / defaultArm;
#             ['LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand'].forEach(bone => {
#               if (boneRefs.current[bone]) boneRefs.current[bone]!.scale.y = armScale;
#             });
#           }
#           if (userLeg) {
#             const legScale = userLeg / defaultLeg;
#             ['LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot'].forEach(bone => {
#               if (boneRefs.current[bone]) boneRefs.current[bone]!.scale.y = legScale;
#             });
#           }
#         };
#         applyMorphing();
#       }
#     }
#     // eslint-disable-next-line
#   }, [measurements]);

#   // Mouse interaction
#   const handleMouseDown = (e: React.MouseEvent) => {
#     setIsDragging(true);
#     setLastMousePos({ x: e.clientX, y: e.clientY });
#   };
#   const handleMouseMove = (e: React.MouseEvent) => {
#     if (!isDragging) return;
#     const deltaX = e.clientX - lastMousePos.x;
#     const deltaY = e.clientY - lastMousePos.y;
#     setRotation(prev => ({ x: prev.x + deltaY, y: prev.y + deltaX }));
#     setLastMousePos({ x: e.clientX, y: e.clientY });
#   };
#   const handleMouseUp = () => setIsDragging(false);


#   // Project mesh center to 2D for tooltip
#   function getMeshScreenPosition(meshNames: string[]): { x: number; y: number } | null {
#     if (!mountRef.current || !modelRef.current) return null;
#     const rendererRect = mountRef.current.getBoundingClientRect();
#     let mesh: THREE.Mesh | undefined;
#     for (const name of meshNames) {
#       mesh = meshRefs.current[name];
#       if (mesh) break;
#     }
#     if (!mesh) return null;
#     // Get mesh center in world coordinates
#     const box = new THREE.Box3().setFromObject(mesh);
#     const center = new THREE.Vector3();
#     box.getCenter(center);
#     // Project to normalized device coordinates
#     const width = mountRef.current.offsetWidth;
#     const height = mountRef.current.offsetHeight;
#     // Find the camera (assume only one)
#     let camera: THREE.Camera | undefined;
#     for (const child of modelRef.current.parent?.children || []) {
#       if ((child as any).isCamera) camera = child as THREE.Camera;
#     }
#     // Fallback: use a default camera
#     if (!camera) {
#       camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
#       camera.position.set(0, 0, 4);
#     }
#     // Project
#     const vector = center.clone().project(camera);
#     const x = ((vector.x + 1) / 2) * width;
#     const y = ((-vector.y + 1) / 2) * height;
#     return { x, y };
#   }

#   return (
#     <div
#       className={`relative bg-neutral-50 rounded-lg overflow-hidden ${className}`}
#       style={{ minHeight: '600px' }}
#       onMouseLeave={() => { setHighlighted(null); onHighlightMeasurement && onHighlightMeasurement(null); }}
#     >
#       {/* Three.js mount point */}
#       <div
#         ref={mountRef}
#         className="w-full h-full cursor-grab active:cursor-grabbing"
#         style={{ minHeight: '500px' }}
#         onMouseDown={handleMouseDown}
#         onMouseMove={handleMouseMove}
#         onMouseUp={handleMouseUp}
#         onMouseLeave={handleMouseUp}
#       />
#       {/* Tooltip (now projected to mesh center) */}
#       {activeMeasurement && (() => {
#         const m = measurements.measurements.find(m => m.id === activeMeasurement || m.type === activeMeasurement);
#         if (!m) return null;
#         const meshNames = measurementToMesh[m.type] || [];
#         const pos = getMeshScreenPosition(meshNames);
#         // Fallback: show tooltip at top center if projection fails
#         if (!pos) {
#           return (
#             <div
#               className="absolute top-8 left-1/2 -translate-x-1/2 z-20 bg-white border border-yellow-300 rounded-lg shadow-lg p-3 text-sm min-w-[180px]"
#               style={{ filter: 'drop-shadow(0 4px 16px #0002)' }}
#             >
#               <div className="font-semibold text-yellow-700 mb-1">{m.name}</div>
#               <div className="mb-1">Value: <span className="font-bold">{m.value.toFixed(1)}{m.unit}</span></div>
#               <div className="mb-1">Confidence: <span className="font-bold">{Math.round(m.confidence * 100)}%</span></div>
#             </div>
#           );
#         }
#         return (
#           <div
#             className="absolute z-20 bg-white border border-yellow-300 rounded-lg shadow-lg p-3 text-sm min-w-[180px] animate-fade-in"
#             style={{ left: pos.x, top: pos.y, transform: 'translate(-50%, -110%)', filter: 'drop-shadow(0 4px 16px #0002)' }}
#           >
#             <div className="font-semibold text-yellow-700 mb-1">{m.name}</div>
#             <div className="mb-1">Value: <span className="font-bold">{m.value.toFixed(1)}{m.unit}</span></div>
#             <div className="mb-1">Confidence: <span className="font-bold">{Math.round(m.confidence * 100)}%</span></div>
#           </div>
#         );
#       })()}
#       {/* Instructions */}
#       <div className="absolute bottom-4 left-4 text-xs text-neutral-500 bg-white/90 backdrop-blur-sm px-2 py-1 rounded">
#         Drag to rotate • Click reset to center
#       </div>
#     </div>
#   );
# };