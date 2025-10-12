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

    # Confidence calculation helper
    def avg_visibility(keys):
        return round(sum(landmarks[k].visibility for k in keys) / len(keys), 2)

    # Key indices for each measurement
    chest_keys = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    waist_keys = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_keys = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]

    measurements["chest_circumference"] = {"cm": chest_circumference, "inches": cm_to_inches(chest_circumference), "confidence": avg_visibility(chest_keys), "landmarks": chest_keys}
    measurements["waist_circumference"] = {"cm": waist_circumference, "inches": cm_to_inches(waist_circumference), "confidence": avg_visibility(waist_keys), "landmarks": waist_keys}
    measurements["hip_circumference"] = {"cm": hip_circumference, "inches": cm_to_inches(hip_circumference), "confidence": avg_visibility(hip_keys), "landmarks": hip_keys}
    
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

    sleeve_keys = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value]
    measurements["short_sleeve_length"] = {"cm": round(short_sleeve_cm, 1), "inches": cm_to_inches(short_sleeve_cm), "confidence": avg_visibility(sleeve_keys), "landmarks": sleeve_keys}
    measurements["three_quarter_sleeve"] = {"cm": round(three_quarter_sleeve_cm, 1), "inches": cm_to_inches(three_quarter_sleeve_cm), "confidence": avg_visibility(sleeve_keys), "landmarks": sleeve_keys}
    measurements["long_sleeve_length"] = {"cm": round(long_sleeve_cm, 1), "inches": cm_to_inches(long_sleeve_cm), "confidence": avg_visibility(sleeve_keys), "landmarks": sleeve_keys}

    # Biceps circumference estimate
    biceps_cm = clamp_measurement(pixel_to_cm(upper_arm_px) * 0.7, 0.07, 0.25, user_height_cm)
    biceps_keys = [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    measurements["biceps_circumference"] = {"cm": round(biceps_cm, 1), "inches": cm_to_inches(biceps_cm), "confidence": avg_visibility(biceps_keys), "landmarks": biceps_keys}

    # Lower body
    inseam_px = calculate_distance(left_hip, left_ankle)
    thigh_px = calculate_distance(left_hip, left_knee)

    inseam_cm = pixel_to_cm(inseam_px)
    thigh_circumference_cm = pixel_to_cm(thigh_px * 0.6)
    inseam_cm = clamp_measurement(inseam_cm, 0.35, 0.55, user_height_cm)
    thigh_circumference_cm = clamp_measurement(thigh_circumference_cm, 0.12, 0.32, user_height_cm)

    inseam_keys = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value]
    thigh_keys = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value]
    measurements["inseam"] = {"cm": round(inseam_cm, 1), "inches": cm_to_inches(inseam_cm), "confidence": avg_visibility(inseam_keys), "landmarks": inseam_keys}
    measurements["thigh_circumference"] = {"cm": round(thigh_circumference_cm, 1), "inches": cm_to_inches(thigh_circumference_cm), "confidence": avg_visibility(thigh_keys), "landmarks": thigh_keys}

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

    top_keys = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
    full_keys = top_keys + [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_HEEL.value]
    measurements["top_length"] = {"cm": round(top_length_cm, 1), "inches": cm_to_inches(top_length_cm), "confidence": avg_visibility(top_keys), "landmarks": top_keys}
    measurements["full_length"] = {"cm": round(full_length_cm, 1), "inches": cm_to_inches(full_length_cm), "confidence": avg_visibility(full_keys), "landmarks": full_keys}
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

    def avg_visibility(keys):
        return round(sum(landmarks[k].visibility for k in keys) / len(keys), 2)

    bust_keys = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    waist_keys = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_keys = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]

    measurements["bust_circumference"] = {"cm": bust_circumference, "inches": cm_to_inches(bust_circumference), "confidence": avg_visibility(bust_keys), "landmarks": bust_keys}
    measurements["waist_circumference"] = {"cm": waist_circumference, "inches": cm_to_inches(waist_circumference), "confidence": avg_visibility(waist_keys), "landmarks": waist_keys}
    measurements["hip_circumference"] = {"cm": hip_circumference, "inches": cm_to_inches(hip_circumference), "confidence": avg_visibility(hip_keys), "landmarks": hip_keys}

    # Sleeves
    upper_arm_px = calculate_distance(left_shoulder, left_elbow)
    full_arm_px = calculate_distance(left_shoulder, left_wrist)

    short_sleeve_cm = pixel_to_cm(upper_arm_px)
    three_quarter_sleeve_cm = pixel_to_cm(upper_arm_px * 1.5)
    long_sleeve_cm = pixel_to_cm(full_arm_px)
    
    short_sleeve_cm = clamp_measurement(short_sleeve_cm, 0.12, 0.30, user_height_cm)
    three_quarter_sleeve_cm = clamp_measurement(three_quarter_sleeve_cm, 0.20, 0.45, user_height_cm)
    long_sleeve_cm = clamp_measurement(long_sleeve_cm, 0.35, 0.6, user_height_cm)

    sleeve_keys = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value]
    measurements["short_sleeve_length"] = {"cm": round(short_sleeve_cm, 1), "inches": cm_to_inches(short_sleeve_cm), "confidence": avg_visibility(sleeve_keys), "landmarks": sleeve_keys}
    measurements["three_quarter_sleeve"] = {"cm": round(three_quarter_sleeve_cm, 1), "inches": cm_to_inches(three_quarter_sleeve_cm), "confidence": avg_visibility(sleeve_keys), "landmarks": sleeve_keys}
    measurements["long_sleeve_length"] = {"cm": round(long_sleeve_cm, 1), "inches": cm_to_inches(long_sleeve_cm), "confidence": avg_visibility(sleeve_keys), "landmarks": sleeve_keys}

    # Biceps
    biceps_cm = clamp_measurement(pixel_to_cm(upper_arm_px) * 0.6, 0.07, 0.25, user_height_cm)
    biceps_keys = [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    measurements["biceps_circumference"] = {"cm": round(biceps_cm, 1), "inches": cm_to_inches(biceps_cm), "confidence": avg_visibility(biceps_keys), "landmarks": biceps_keys}

    # Lower body
    left_hip_l = left_hip
    inseam_px = calculate_distance(left_hip_l, left_ankle)
    thigh_px = calculate_distance(left_hip_l, left_knee)

    inseam_cm = pixel_to_cm(inseam_px)
    thigh_circumference_cm = pixel_to_cm(thigh_px * 0.6)

    inseam_cm = clamp_measurement(inseam_cm, 0.35, 0.55, user_height_cm)
    thigh_circumference_cm = clamp_measurement(thigh_circumference_cm, 0.12, 0.32, user_height_cm)

    inseam_keys = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value]
    thigh_keys = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value]
    measurements["inseam"] = {"cm": round(inseam_cm, 1), "inches": cm_to_inches(inseam_cm), "confidence": avg_visibility(inseam_keys), "landmarks": inseam_keys}
    measurements["thigh_circumference"] = {"cm": round(thigh_circumference_cm, 1), "inches": cm_to_inches(thigh_circumference_cm), "confidence": avg_visibility(thigh_keys), "landmarks": thigh_keys}

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

    top_keys = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
    full_keys = top_keys + [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_HEEL.value]
    measurements["top_length"] = {"cm": round(top_length_cm, 1), "inches": cm_to_inches(top_length_cm), "confidence": avg_visibility(top_keys), "landmarks": top_keys}
    measurements["full_length"] = {"cm": round(full_length_cm, 1), "inches": cm_to_inches(full_length_cm), "confidence": avg_visibility(full_keys), "landmarks": full_keys}
    return measurements









# Placeholder for process_images_and_get_measurements

def process_images_and_get_measurements(front_frame, side_frame, gender, user_height_cm):
    # Move the full implementation from app.py
    pass

# Example stub for get_user_measurements (replace with full logic):

from flask_jwt_extended import get_jwt_identity, get_jwt

def get_user_measurements(request):
    user_id = get_jwt_identity()
    # claims = get_jwt()
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
    from flask_jwt_extended import get_jwt_identity, get_jwt

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

        # Get user info from JWT
        user_id = get_jwt_identity()

        # Store images in GridFS
        front_image_id = fs.put(front_image_data, filename=front_image_file.filename, content_type=front_image_file.content_type)
        side_image_id = fs.put(side_image_data, filename=side_image_file.filename, content_type=side_image_file.content_type)

        # Store measurement record
        record = {
            'user_id': user_id,
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

        return jsonify({'record': record}), 201
    except Exception as e:
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500


# Stub for get_measurement_record (replace with full logic)


def get_measurement_record(record_id):
    """
    Fetch a single measurement record by ID, check user permissions, and return JSON.
    """
    from flask_jwt_extended import get_jwt_identity, get_jwt
    try:
        record = measurements_coll.find_one({'_id': ObjectId(record_id)})
        if not record:
            return jsonify({'error': 'Measurement record not found.'}), 404
        user_id = get_jwt_identity()
        if record.get('user_id') != user_id:
            return jsonify({'error': 'Unauthorized access.'}), 403
        record['_id'] = str(record['_id'])
        if 'front_image_id' in record:
            record['front_image_id'] = str(record['front_image_id'])
        if 'side_image_id' in record:
            record['side_image_id'] = str(record['side_image_id'])
        return jsonify({'record': record}), 200
    except Exception as e:
        return jsonify({'error': f'Error fetching record: {str(e)}'}), 500

# Stub for get_image (replace with full logic)

def get_image(image_id):
    """
    Fetch and stream image from GridFS by image_id with correct content type.
    """
    from flask import Response
    try:
        grid_out = fs.get(ObjectId(image_id))
        response = Response(grid_out.read(), mimetype=grid_out.content_type)
        response.headers.set('Content-Disposition', 'inline', filename=grid_out.filename)
        return response
    except Exception as e:
        return jsonify({'error': f'Error fetching image: {str(e)}'}), 404
