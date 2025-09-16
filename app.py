import cv2
import numpy as np
import mediapipe as mp
import torch
from flask import Flask, request, jsonify, render_template_string
import torch.nn.functional as F
import base64
import io
from PIL import Image

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
pose = mp_pose.Pose(model_complexity=2)  # Improved accuracy
holistic = mp_holistic.Holistic()  # For refining pose

KNOWN_OBJECT_WIDTH_CM = 21.0  # A4 paper width in cm
FOCAL_LENGTH = 600  # Default focal length
DEFAULT_HEIGHT_CM = 152.0  # Default height if not provided


# Load depth estimation model
def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model


depth_model = load_depth_model()


def calibrate_focal_length(image, real_width_cm, detected_width_px):
    """Dynamically calibrates focal length using a known object."""
    return (detected_width_px * FOCAL_LENGTH) / real_width_cm if detected_width_px else FOCAL_LENGTH


def detect_reference_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        focal_length = calibrate_focal_length(image, KNOWN_OBJECT_WIDTH_CM, w)
        scale_factor = KNOWN_OBJECT_WIDTH_CM / w
        return scale_factor, focal_length
    return 0.05, FOCAL_LENGTH


def estimate_depth(image):
    """Uses AI-based depth estimation to improve circumference calculations."""
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Resize input to match MiDaS model input size
    input_tensor = F.interpolate(input_tensor, size=(384, 384), mode="bilinear", align_corners=False)

    with torch.no_grad():
        depth_map = depth_model(input_tensor)

    return depth_map.squeeze().numpy()


def calculate_distance_using_height(landmarks, image_height, user_height_cm):
    """Calculate distance using the user's known height."""
    top_head = landmarks[mp_pose.PoseLandmark.NOSE.value].y * image_height
    bottom_foot = max(
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    ) * image_height

    person_height_px = abs(bottom_foot - top_head)

    # Using the formula: distance = (actual_height_cm * focal_length) / height_in_pixels
    distance = (user_height_cm * FOCAL_LENGTH) / person_height_px

    # Calculate more accurate scale_factor based on known height
    scale_factor = user_height_cm / person_height_px

    return distance, scale_factor


def get_body_width_at_height(frame, height_px, center_x):
    """Scan horizontally at a specific height to find body edges."""
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    # Ensure height_px is within image bounds
    if height_px >= frame.shape[0]:
        height_px = frame.shape[0] - 1

    # Get horizontal line at the specified height
    horizontal_line = thresh[height_px, :]

    # Find left and right edges starting from center
    center_x = int(center_x * frame.shape[1])
    left_edge, right_edge = center_x, center_x

    # Scan from center to left
    for i in range(center_x, 0, -1):
        if horizontal_line[i] == 0:  # Found edge (black pixel)
            left_edge = i
            break

    # Scan from center to right
    for i in range(center_x, len(horizontal_line)):
        if horizontal_line[i] == 0:  # Found edge (black pixel)
            right_edge = i
            break

    width_px = right_edge - left_edge

    # If width is unreasonably small, apply a minimum width
    min_width = 0.1 * frame.shape[1]  # Minimum width as 10% of image width
    if width_px < min_width:
        width_px = min_width

    return width_px


def calculate_measurements(results, scale_factor, image_width, image_height, depth_map, frame=None,
                           user_height_cm=None):
    landmarks = results.pose_landmarks.landmark

    # If user's height is provided, use it to get a more accurate scale factor
    if user_height_cm:
        _, scale_factor = calculate_distance_using_height(landmarks, image_height, user_height_cm)

    def pixel_to_cm(value):
        return round(value * scale_factor, 2)

    def calculate_circumference(width_px, depth_ratio=1.0):
        """Estimate circumference using width and depth adjustment."""
        # Using a simplified elliptical approximation: C ≈ 2π * sqrt((a² + b²)/2)
        # where a is half the width and b is estimated depth
        width_cm = width_px * scale_factor
        estimated_depth_cm = width_cm * depth_ratio * 0.7  # Depth is typically ~70% of width for torso
        half_width = width_cm / 2
        half_depth = estimated_depth_cm / 2
        return round(2 * np.pi * np.sqrt((half_width ** 2 + half_depth ** 2) / 2), 2)

    measurements = {}

    # Shoulder Width
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_width_px = abs(left_shoulder.x * image_width - right_shoulder.x * image_width)

    # Apply a slight correction factor for shoulders (they're usually detected well)
    shoulder_correction = 1.1  # 10% wider
    shoulder_width_px *= shoulder_correction

    measurements["shoulder_width"] = pixel_to_cm(shoulder_width_px)

    # Chest/Bust Measurement
    chest_y_ratio = 0.15  # Approximately 15% down from shoulder to hip
    chest_y = left_shoulder.y + (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y - left_shoulder.y) * chest_y_ratio

    chest_correction = 1.15  # 15% wider than detected width
    chest_width_px = abs((right_shoulder.x - left_shoulder.x) * image_width) * chest_correction

    if frame is not None:
        chest_y_px = int(chest_y * image_height)
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        detected_width = get_body_width_at_height(frame, chest_y_px, center_x)
        if detected_width > 0:
            chest_width_px = max(chest_width_px, detected_width)

    chest_depth_ratio = 1.0
    if depth_map is not None:
        chest_x = int(((left_shoulder.x + right_shoulder.x) / 2) * image_width)
        chest_y_px = int(chest_y * image_height)
        scale_y = 384 / image_height
        scale_x = 384 / image_width
        chest_y_scaled = int(chest_y_px * scale_y)
        chest_x_scaled = int(chest_x * scale_x)
        if 0 <= chest_y_scaled < 384 and 0 <= chest_x_scaled < 384:
            chest_depth = depth_map[chest_y_scaled, chest_x_scaled]
            max_depth = np.max(depth_map)
            chest_depth_ratio = 1.0 + 0.5 * (1.0 - chest_depth / max_depth)

    measurements["chest_width"] = pixel_to_cm(chest_width_px)
    measurements["chest_circumference"] = calculate_circumference(chest_width_px, chest_depth_ratio)

    # Waist Measurement
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Adjust waist_y_ratio to better reflect the natural waistline
    waist_y_ratio = 0.35  # 35% down from shoulder to hip (higher than before)
    waist_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * waist_y_ratio

    # Use contour detection to dynamically estimate waist width
    if frame is not None:
        waist_y_px = int(waist_y * image_height)
        center_x = (left_hip.x + right_hip.x) / 2
        detected_width = get_body_width_at_height(frame, waist_y_px, center_x)
        if detected_width > 0:
            waist_width_px = detected_width
        else:
            # Fallback to hip width if contour detection fails
            waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9  # 90% of hip width
    else:
        # Fallback to hip width if no frame is provided
        waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9  # 90% of hip width

    # Apply 30% correction factor to waist width
    waist_correction = 1.16  # 30% wider
    waist_width_px *= waist_correction

    # Get depth adjustment for waist if available
    waist_depth_ratio = 1.0
    if depth_map is not None:
        waist_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
        waist_y_px = int(waist_y * image_height)
        scale_y = 384 / image_height
        scale_x = 384 / image_width
        waist_y_scaled = int(waist_y_px * scale_y)
        waist_x_scaled = int(waist_x * scale_x)
        if 0 <= waist_y_scaled < 384 and 0 <= waist_x_scaled < 384:
            waist_depth = depth_map[waist_y_scaled, waist_x_scaled]
            max_depth = np.max(depth_map)
            waist_depth_ratio = 1.0 + 0.5 * (1.0 - waist_depth / max_depth)

    measurements["waist_width"] = pixel_to_cm(waist_width_px)
    measurements["waist"] = calculate_circumference(waist_width_px, waist_depth_ratio)
    # Hip Measurement
    hip_correction = 1.35  # Hips are typically 35% wider than detected landmarks
    hip_width_px = abs(left_hip.x * image_width - right_hip.x * image_width) * hip_correction

    if frame is not None:
        hip_y_offset = 0.1  # 10% down from hip landmarks
        hip_y = left_hip.y + (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y - left_hip.y) * hip_y_offset
        hip_y_px = int(hip_y * image_height)
        center_x = (left_hip.x + right_hip.x) / 2
        detected_width = get_body_width_at_height(frame, hip_y_px, center_x)
        if detected_width > 0:
            hip_width_px = max(hip_width_px, detected_width)

    hip_depth_ratio = 1.0
    if depth_map is not None:
        hip_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
        hip_y_px = int(left_hip.y * image_height)
        hip_y_scaled = int(hip_y_px * scale_y)
        hip_x_scaled = int(hip_x * scale_x)
        if 0 <= hip_y_scaled < 384 and 0 <= hip_x_scaled < 384:
            hip_depth = depth_map[hip_y_scaled, hip_x_scaled]
            max_depth = np.max(depth_map)
            hip_depth_ratio = 1.0 + 0.5 * (1.0 - hip_depth / max_depth)

    measurements["hip_width"] = pixel_to_cm(hip_width_px)
    measurements["hip"] = calculate_circumference(hip_width_px, hip_depth_ratio)

    # Other measurements (unchanged)
    neck = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    neck_width_px = abs(neck.x * image_width - left_ear.x * image_width) * 2.0
    measurements["neck"] = calculate_circumference(neck_width_px, 1.0)
    measurements["neck_width"] = pixel_to_cm(neck_width_px)

    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    sleeve_length_px = abs(left_shoulder.y * image_height - left_wrist.y * image_height)
    measurements["arm_length"] = pixel_to_cm(sleeve_length_px)

    shirt_length_px = abs(left_shoulder.y * image_height - left_hip.y * image_height) * 1.2
    measurements["shirt_length"] = pixel_to_cm(shirt_length_px)

    # Thigh Circumference (improved with depth information)
    thigh_y_ratio = 0.2  # 20% down from hip to knee
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    thigh_y = left_hip.y + (left_knee.y - left_hip.y) * thigh_y_ratio

    # Apply correction factor for thigh width
    thigh_correction = 1.2  # Thighs are typically wider than what can be estimated from front view
    thigh_width_px = hip_width_px * 0.5 * thigh_correction  # Base thigh width on hip width

    # Use contour detection if frame is available
    if frame is not None:
        thigh_y_px = int(thigh_y * image_height)
        thigh_x = left_hip.x * 0.9  # Move slightly inward from hip
        detected_width = get_body_width_at_height(frame, thigh_y_px, thigh_x)
        if detected_width > 0 and detected_width < hip_width_px:  # Sanity check
            thigh_width_px = detected_width  # Use detected width

    # If depth map is available, use it for thigh measurement
    thigh_depth_ratio = 1.0
    if depth_map is not None:
        thigh_x = int(left_hip.x * image_width)
        thigh_y_px = int(thigh_y * image_height)

        # Scale coordinates to match depth map size
        thigh_y_scaled = int(thigh_y_px * scale_y)
        thigh_x_scaled = int(thigh_x * scale_x)

        if 0 <= thigh_y_scaled < 384 and 0 <= thigh_x_scaled < 384:
            thigh_depth = depth_map[thigh_y_scaled, thigh_x_scaled]
            max_depth = np.max(depth_map)
            thigh_depth_ratio = 1.0 + 0.5 * (1.0 - thigh_depth / max_depth)

    measurements["thigh"] = pixel_to_cm(thigh_width_px)
    measurements["thigh_circumference"] = calculate_circumference(thigh_width_px, thigh_depth_ratio)

    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    trouser_length_px = abs(left_hip.y * image_height - left_ankle.y * image_height)
    measurements["trouser_length"] = pixel_to_cm(trouser_length_px)

    return measurements


def validate_front_image(image_np):
    """
    Basic validation for front image to ensure:
    - There is a person in the image
    - Not just a face/selfie (upper body visible)
    - Key upper landmarks are detected
    """
    try:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_height, image_width = image_np.shape[:2]

        # Process with MediaPipe Holistic
        with mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                refine_face_landmarks=False) as holistic:

            results = holistic.process(rgb_frame)

        if not hasattr(results, 'pose_landmarks') or not results.pose_landmarks:
            return False, "No person detected. Please make sure you're clearly visible in the frame."

        # Minimum required upper body landmarks
        MINIMUM_LANDMARKS = [
            mp_holistic.PoseLandmark.NOSE,
            mp_holistic.PoseLandmark.LEFT_SHOULDER,
            mp_holistic.PoseLandmark.RIGHT_SHOULDER,
            mp_holistic.PoseLandmark.LEFT_ELBOW,
            mp_holistic.PoseLandmark.RIGHT_ELBOW,
            mp_holistic.PoseLandmark.RIGHT_KNEE,
            mp_holistic.PoseLandmark.LEFT_KNEE

        ]

        # Verify minimum landmarks are detected
        missing_upper = []
        for landmark in MINIMUM_LANDMARKS:
            landmark_data = results.pose_landmarks.landmark[landmark]
            if (landmark_data.visibility < 0.5 or
                    landmark_data.x < 0 or
                    landmark_data.x > 1 or
                    landmark_data.y < 0 or
                    landmark_data.y > 1):
                missing_upper.append(landmark.name.replace('_', ' '))

        if missing_upper:
            return False, f"Couldn't detect full body. Please make sure your full body is visible."

        # Check if this might be just a face/selfie (no torso)
        nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

        # Calculate approximate upper body size
        shoulder_width = abs(left_shoulder.x - right_shoulder.x) * image_width
        head_to_shoulder = abs(left_shoulder.y - nose.y) * image_height

        # If the shoulder width is small compared to head size, likely a selfie
        if shoulder_width < head_to_shoulder * 1.2:
            return False, "Please step back to show more of your upper body, not just your face."

        return True, "Validation passed - proceeding with measurements"

    except Exception as e:
        print(f"Error validating body image: {e}")
        return False, "You arent providing images correctly. Please try again."


# Updated HTML with Feet+Inches OR Centimeters selector
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Body Measurement Tool</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="file"], input[type="number"], select { width: 100%; padding: 8px; margin-bottom: 5px; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .results { margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <h1>Body Measurement Tool</h1>
    <form action="/upload_images" method="post" enctype="multipart/form-data">
        <div class="form-group">
            <label for="front">Front Image (required):</label>
            <input type="file" id="front" name="front" accept="image/*" required>
        </div>

        <div class="form-group">
            <label for="left_side">Left Side Image:</label>
            <input type="file" id="left_side" name="left_side" accept="image/*">
        </div>

        <div class="form-group">
            <label>Your Height:</label>
            <select id="height_unit" name="height_unit" onchange="toggleHeightInputs()">
                <option value="ft">Feet + Inches</option>
                <option value="cm">Centimeters</option>
            </select>

            <!-- Feet + Inches -->
            <div id="feet_inches_input">
                <input type="number" id="height_ft" name="height_ft" min="0" placeholder="Feet">
                <input type="number" id="height_in" name="height_in" min="0" max="11" placeholder="Inches">
            </div>

            <!-- Centimeters -->
            <div id="cm_input" style="display:none;">
                <input type="number" id="height_cm" name="height_cm" min="0" placeholder="Centimeters">
            </div>
        </div>

        <button type="submit">Calculate Measurements</button>
    </form>

    {% if message %}
    <div class="{% if message_type == 'error' %}error{% else %}success{% endif %}">
        {{ message }}
    </div>
    {% endif %}

    {% if measurements %}
    <div class="results">
        <h2>Measurement Results</h2>
        <table>
            {% for key, value in measurements.items() %}
            <tr>
                <td><strong>{{ key.replace('_', ' ').title() }}:</strong></td>
                <td>{{ value }} cm ({{ (value * 0.393701) | round(2) }} in)</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    <script>
    function toggleHeightInputs() {
        const unit = document.getElementById("height_unit").value;
        if (unit === "ft") {
            document.getElementById("feet_inches_input").style.display = "block";
            document.getElementById("cm_input").style.display = "none";
        } else {
            document.getElementById("feet_inches_input").style.display = "none";
            document.getElementById("cm_input").style.display = "block";
        }
    }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/upload_images", methods=["GET", "POST"])
def upload_images():
    if request.method == "GET":
        return render_template_string(HTML_TEMPLATE)

    if "front" not in request.files:
        return render_template_string(HTML_TEMPLATE, message="Missing front image for reference.", message_type="error")

    front_image_file = request.files["front"]
    if front_image_file.filename == '':
        return render_template_string(HTML_TEMPLATE, message="No file selected for front image.", message_type="error")

    front_image_np = np.frombuffer(front_image_file.read(), np.uint8)
    front_image_file.seek(0)

    is_valid, error_msg = validate_front_image(cv2.imdecode(front_image_np, cv2.IMREAD_COLOR))
    if not is_valid:
        return render_template_string(HTML_TEMPLATE, message=error_msg, message_type="error")

    # ---- Handle height input (feet+inches OR cm) ----
    height_unit = request.form.get("height_unit", "cm")
    try:
        if height_unit == "ft":
            height_ft = float(request.form.get("height_ft") or 0)
            height_in = float(request.form.get("height_in") or 0)
            user_height_cm = (height_ft * 30.48) + (height_in * 2.54)
        else:  # cm
            height_cm = float(request.form.get("height_cm") or 0)
            user_height_cm = height_cm

        if user_height_cm == 0:
            user_height_cm = DEFAULT_HEIGHT_CM
    except ValueError:
        user_height_cm = DEFAULT_HEIGHT_CM
    # -----------------------------------------------

    received_images = {pose_name: request.files[pose_name] for pose_name in ["front", "left_side"]
                       if pose_name in request.files and request.files[pose_name].filename != ''}

    measurements, scale_factor, focal_length, results = {}, None, FOCAL_LENGTH, {}
    frames = {}

    for pose_name, image_file in received_images.items():
        image_np = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        frames[pose_name] = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results[pose_name] = holistic.process(rgb_frame)
        image_height, image_width, _ = frame.shape

        if pose_name == "front":
            if results[pose_name].pose_landmarks:
                _, scale_factor = calculate_distance_using_height(
                    results[pose_name].pose_landmarks.landmark,
                    image_height,
                    user_height_cm
                )
            else:
                scale_factor, focal_length = detect_reference_object(frame)

        depth_map = estimate_depth(frame) if pose_name in ["front", "left_side"] else None

        if results[pose_name].pose_landmarks:
            if pose_name == "front":
                measurements.update(calculate_measurements(
                    results[pose_name],
                    scale_factor,
                    image_width,
                    image_height,
                    depth_map,
                    frames[pose_name],
                    user_height_cm
                ))

    return render_template_string(
        HTML_TEMPLATE,
        measurements=measurements,
        message="Measurements calculated successfully!",
        message_type="success"
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)