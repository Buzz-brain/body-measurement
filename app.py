import cv2
import numpy as np
import mediapipe as mp
import torch
from flask import Flask, request, jsonify, render_template_string
import torch.nn.functional as F
import base64
import io
from PIL import Image
import math

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
pose = mp_pose.Pose(model_complexity=2)
holistic = mp_holistic.Holistic()

KNOWN_OBJECT_WIDTH_CM = 21.0
FOCAL_LENGTH = 600
DEFAULT_HEIGHT_CM = 170.0


# Load depth estimation model
def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model


depth_model = load_depth_model()


# --- small helpers ---
def cm_to_inches(cm):
    return round(cm * 0.393701, 2)


def clamp_measurement(value_cm, min_ratio, max_ratio, user_height_cm):
    """Clamp a measurement in cm to a percentage range of user height."""
    min_val = user_height_cm * min_ratio
    max_val = user_height_cm * max_ratio
    return round(max(min_val, min(value_cm, max_val)), 2)


# === HELPER FUNCTIONS ===

def calculate_effective_scale(landmarks, image_width, image_height, known_height_cm):
    """
    Blend height-based and torso-based scale factors for more accurate cm/px conversion
    Returns (effective_scale_cm_per_pixel, pixel_full_height)
    """
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
        # fallback
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
    measurements["thigh_circumference"] = {"cm": round(thigh_circumference_cm, 1), "inches": cm_to_inches(thigh_circumference_cm)}

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


# === HTML TEMPLATE ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Professional Body Measurement Tool</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; text-align: center; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="file"], input[type="number"], select { width: 100%; padding: 8px; margin-bottom: 5px; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .results { margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }
        .error { color: red; }
        .success { color: green; }
        .instructions { background-color: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .measurement-category { background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .measurement-item { display: flex; justify-content: space-between; margin: 5px 0; }
    </style>
</head>
<body>
    <h1>Professional Body Measurement System</h1>

    <div class="instructions">
        <h3>📋 Instructions for Accurate Measurements:</h3>
        <ul>
            <li>Stand with feet shoulder-width apart</li>
            <li>Keep arms slightly away from your body</li>
            <li>Wear fitted clothing for accurate measurements</li>
            <li>Ensure good lighting without strong shadows</li>
            <li>Stand directly facing the camera for front view</li>
            <li>Turn 90 degrees for side view</li>
        </ul>
    </div>

    <form action="/upload_images" method="post" enctype="multipart/form-data">
        <!-- Gender Selection -->
        <div class="form-group">
            <label for="gender">Select Gender:</label>
            <select id="gender" name="gender" required>
                <option value="male">Male</option>
                <option value="female">Female</option>
            </select>
        </div>

        <!-- Front and Side Images -->
        <div class="form-group">
            <label for="front">Front View Image (required):</label>
            <input type="file" id="front" name="front" accept="image/*" required>
        </div>

        <div class="form-group">
            <label for="side">Side View Image (required):</label>
            <input type="file" id="side" name="side" accept="image/*" required>
        </div>

        <!-- Height Input -->
        <div class="form-group">
            <label>Your Height:</label>
            <select id="height_unit" name="height_unit" onchange="toggleHeightInputs()">
                <option value="ft">Feet + Inches</option>
                <option value="cm">Centimeters</option>
            </select>

            <div id="feet_inches_input">
                <input type="number" id="height_ft" name="height_ft" min="0" placeholder="Feet" required>
                <input type="number" id="height_in" name="height_in" min="0" max="11" placeholder="Inches" required>
            </div>

            <div id="cm_input" style="display:none;">
                <input type="number" id="height_cm" name="height_cm" min="0" placeholder="Centimeters" disabled>
            </div>
        </div>

        <button type="submit">Generate Professional Measurements</button>
    </form>

    {% if message %}
    <div class="{% if message_type == 'error' %}error{% else %}success{% endif %}">
        {{ message }}
    </div>
    {% endif %}

    {% if measurements %}
    <div class="results">
        <h2>📏 Professional Measurement Results</h2>
        <p><strong>Gender:</strong> {{ gender|title }}</p>
        <p><strong>Estimated Height:</strong> {{ measurements.estimated_height.cm }} cm ({{ measurements.estimated_height.inches }} inches)</p>

        <div class="measurement-category">
            <h3>📐 Upper Body Measurements</h3>
            {% for key in ['head_circumference', 'neck_circumference', 'shoulder_width'] %}
            {% if measurements[key] %}
            <div class="measurement-item">
                <span>{{ key.replace('_', ' ').title() }}:</span>
                <span>{{ measurements[key].cm }} cm ({{ measurements[key].inches }} in)</span>
            </div>
            {% endif %}
            {% endfor %}
        </div>

        <div class="measurement-category">
            <h3>👕 Torso Measurements</h3>
            {% for key in ['bust_circumference', 'upper_bust', 'chest_circumference', 'waist_circumference', 'hip_circumference'] %}
            {% if measurements[key] %}
            <div class="measurement-item">
                <span>{{ key.replace('_', ' ').title() }}:</span>
                <span>{{ measurements[key].cm }} cm ({{ measurements[key].inches }} in)</span>
            </div>
            {% endif %}
            {% endfor %}
        </div>

        <div class="measurement-category">
            <h3>👔 Sleeve Measurements</h3>
            {% for key in ['short_sleeve_length', 'three_quarter_sleeve', 'long_sleeve_length', 'biceps_circumference'] %}
            {% if measurements[key] %}
            <div class="measurement-item">
                <span>{{ key.replace('_', ' ').title() }}:</span>
                <span>{{ measurements[key].cm }} cm ({{ measurements[key].inches }} in)</span>
            </div>
            {% endif %}
            {% endfor %}
        </div>

        <div class="measurement-category">
            <h3>👖 Lower Body Measurements</h3>
            {% for key in ['thigh_circumference', 'inseam'] %}
            {% if measurements[key] %}
            <div class="measurement-item">
                <span>{{ key.replace('_', ' ').title() }}:</span>
                <span>{{ measurements[key].cm }} cm ({{ measurements[key].inches }} in)</span>
            </div>
            {% endif %}
            {% endfor %}
        </div>

        <div class="measurement-category">
            <h3>📏 Length Measurements</h3>
            {% for key in ['top_length', 'full_length'] %}
            {% if measurements[key] %}
            <div class="measurement-item">
                <span>{{ key.replace('_', ' ').title() }}:</span>
                <span>{{ measurements[key].cm }} cm ({{ measurements[key].inches }} in)</span>
            </div>
            {% endif %}
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <script>
    function toggleHeightInputs() {
        const unit = document.getElementById("height_unit").value;
        const ft = document.getElementById("height_ft");
        const inch = document.getElementById("height_in");
        const cm = document.getElementById("height_cm");
        if (unit === "ft") {
            document.getElementById("feet_inches_input").style.display = "block";
            document.getElementById("cm_input").style.display = "none";
            ft.required = true;
            inch.required = true;
            cm.required = false;
            cm.disabled = true;
        } else {
            document.getElementById("feet_inches_input").style.display = "none";
            document.getElementById("cm_input").style.display = "block";
            ft.required = false;
            inch.required = false;
            cm.required = true;
            cm.disabled = false;
        }
    }
    document.addEventListener('DOMContentLoaded', toggleHeightInputs);
    </script>
</body>
</html>
"""


# === ROUTES ===

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/upload_images", methods=["POST"])
def upload_images():
    if "front" not in request.files or "side" not in request.files:
        return render_template_string(HTML_TEMPLATE, message="Both front and side images are required.",
                                      message_type="error")

    front_image_file = request.files["front"]
    side_image_file = request.files["side"]
    gender = request.form.get("gender", "male")

    if front_image_file.filename == '' or side_image_file.filename == '':
        return render_template_string(HTML_TEMPLATE, message="Please select both front and side images.",
                                      message_type="error")

    # Process height input
    height_unit = request.form.get("height_unit", "cm")
    try:
        if height_unit == "ft":
            height_ft = float(request.form.get("height_ft") or 0)
            height_in = float(request.form.get("height_in") or 0)
            user_height_cm = (height_ft * 30.48) + (height_in * 2.54)
        else:
            user_height_cm = float(request.form.get("height_cm") or 0)

        if user_height_cm == 0:
            user_height_cm = DEFAULT_HEIGHT_CM
    except ValueError:
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
            return render_template_string(HTML_TEMPLATE, message=error_msg, message_type="error")

        # MediaPipe
        front_rgb = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
        front_results = holistic.process(front_rgb)

        if not front_results.pose_landmarks:
            return render_template_string(HTML_TEMPLATE, message="Could not detect pose in front image.",
                                          message_type="error")

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

        return render_template_string(
            HTML_TEMPLATE,
            measurements=measurements,
            gender=gender,
            message="Professional measurements generated successfully!",
            message_type="success"
        )

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, message=f"Error processing images: {str(e)}", message_type="error")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
