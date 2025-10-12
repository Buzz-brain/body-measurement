
<div align="center">
  <img src="https://img.icons8.com/color/96/tape-measure.png" width="80" alt="Tape Measure Icon"/>
  
  <h1>AI Body Measurement API</h1>
  <h3>for Tailoring & Fashion E-Commerce</h3>
  <p>
    <img src="https://img.shields.io/badge/Flask-API-blue?logo=flask"/>
    <img src="https://img.shields.io/badge/MediaPipe-Landmarks-orange?logo=google"/>
    <img src="https://img.shields.io/badge/PyTorch-Depth%20AI-red?logo=pytorch"/>
    <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv"/>
    <img src="https://img.shields.io/badge/Deployed%20on-Render-430098?logo=render"/>
  </p>
  <p>📸 Upload <b>front & side pose images</b> and get instant, AI-powered body measurements for fashion, tailoring, and e-commerce.</p>
</div>

---

## ✨ Features

<ul>
  <li>⚡ <b>Real-time</b> image-based body measurement</li>
  <li>🤖 <b>AI-powered depth estimation</b> (MiDaS + PyTorch)</li>
  <li>📏 <b>±2-3 cm accuracy</b> (A4 paper calibration)</li>
  <li>🧩 <b>MediaPipe</b> pose landmark detection</li>
  <li>🔒 <b>Secure REST API</b> (JWT Auth ready)</li>
  <li>🌐 <b>Easy integration</b> with any web/mobile frontend</li>
  <li>🚀 <b>Deployable</b> on <b>Render</b> (backend) & <b>Vercel</b> (frontend)</li>
</ul>

---

## 🛠️ Tech Stack

| <img src="https://img.icons8.com/ios-filled/24/000000/flask.png"/> Flask | <img src="https://img.icons8.com/color/24/000000/opencv.png"/> OpenCV | <img src="https://img.icons8.com/color/24/000000/pytorch.png"/> PyTorch | <img src="https://img.icons8.com/color/24/000000/google-logo.png"/> MediaPipe | <img src="https://img.icons8.com/color/24/000000/render.png"/> Render |
|---|---|---|---|---|

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server (for local dev)
python app.py

# Or for production (Render):
gunicorn app:app
```

---

## 🌍 API Usage

### POST `/measurements`

<details>
<summary>Show Example Request</summary>

```bash
curl -X POST https://your-backend.onrender.com/measurements \
  -F "front_image=@front.jpg" \
  -F "side_image=@side.jpg" \
  -F "user_height_cm=170"
```
</details>

#### Request Fields

| Field           | Type      | Required | Description                                 |
|-----------------|-----------|----------|---------------------------------------------|
| `front_image`   | file      | Yes      | JPEG/PNG image (front pose)                 |
| `side_image`    | file      | No       | JPEG/PNG image (side pose, improves accuracy)|
| `user_height_cm`| number    | Yes      | Real height in centimeters                  |

#### Response

Returns JSON with all measurements, confidence scores, and landmark points used.

---

## 📏 Measurements Provided

| Name                    | Description                                 | Confidence | Points Used |
|-------------------------|---------------------------------------------|------------|-------------|
| `shoulder_width`        | Distance between shoulders                  |    ✅      |     ✔️      |
| `chest_circumference`   | Estimated chest circumference               |    ✅      |     ✔️      |
| `waist_circumference`   | Estimated waist circumference               |    ✅      |     ✔️      |
| `hip_circumference`     | Estimated hip circumference                 |    ✅      |     ✔️      |
| `biceps_circumference`  | Upper arm circumference                     |    ✅      |     ✔️      |
| `thigh_circumference`   | Thigh circumference                        |    ✅      |     ✔️      |
| `inseam`                | Inseam length                              |    ✅      |     ✔️      |
| `long_sleeve_length`    | Long sleeve length                         |    ✅      |     ✔️      |
| `short_sleeve_length`   | Short sleeve length                        |    ✅      |     ✔️      |
| `three_quarter_sleeve`  | 3/4 sleeve length                          |    ✅      |     ✔️      |
| `top_length`            | Top garment length                         |    ✅      |     ✔️      |
| `full_length`           | Full body length                           |    ✅      |     ✔️      |
| `estimated_height`      | Estimated height from image                 |    ✅      |     ✔️      |

---

## �️ Deployment

### Backend (Render)
1. Push your backend code to GitHub.
2. Create a new Web Service on <a href="https://render.com/">Render</a>.
3. Set build command: <code>pip install -r requirements.txt</code>
4. Set start command: <code>gunicorn app:app</code>
5. Add environment variables as needed.
6. Set Flask to listen on <code>0.0.0.0</code> and use <code>PORT</code> env var.

### Frontend (Vercel)
1. Push your frontend (React/Vite) to GitHub.
2. Import your repo on <a href="https://vercel.com/">Vercel</a>.
3. Set project root to <code>frontend</code> folder.
4. Add env var: <code>VITE_API_BASE=https://your-backend.onrender.com</code>
5. Deploy!

---

## 🔗 Integration & UI/UX

<ul>
  <li>🛒 <b>E-commerce</b>: Size suggestions, virtual try-ons</li>
  <li>✂️ <b>Tailoring</b>: Remote client measurements</li>
  <li>🏭 <b>Manufacturers</b>: Personalized size charts</li>
  <li>📱 <b>Fashion apps</b>: Custom-fitted clothing suggestions</li>
</ul>

---

## �‍💻 Contributing

Pull requests and suggestions are welcome! Fork, raise an issue, or open a PR.

---

## 📜 License

MIT License — use freely for personal or commercial projects. Please give credit.



