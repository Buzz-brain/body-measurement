# AI-Based Human Body Measurement System for Tailoring & Fashion E-Commerce

This project is a real-time body measurement API built with **Flask**, **MediaPipe**, **OpenCV**, and **PyTorch**. By analyzing **front and side pose images** of a person, it calculates accurate human body measurements useful for tailoring, clothing size prediction, and virtual fitting rooms.

> 📸 Just send **front and side pose images** (captured using a smartphone or webcam) to this API, and receive key body measurements in centimeters — perfect for fashion retail platforms and tailor-made garment businesses.

---

## Features

- Real-time image-based body measurement
- AI-powered depth estimation using **MiDaS**
- Measurement accuracy with a deviation of **±2-3 cm**
- Calibrates scale using an **A4 paper** as a reference object
- Easily integratable into fashion e-commerce or tailoring platforms
- No external APIs — runs entirely on your local or server environment

---


## Libraries Used

| Library         | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `Flask`        | To expose a simple HTTP API                                             |
| `OpenCV`       | For image processing and contour detection                              |
| `MediaPipe`    | For pose landmark detection (shoulders, hips, etc.)                     |
| `PyTorch`      | For AI-based **depth estimation** using [MiDaS](https://github.com/isl-org/MiDaS) |
| `torchvision`  | Support for model loading & image transformations                       |

---

# How It Works

1. Detects key landmarks using **MediaPipe Pose** (shoulders, hips, knees, ankles).
2. Uses **A4 paper** in the image to calibrate real-world scale from pixels.
3. Enhances width and depth estimation using the **MiDaS depth AI model**.
4. Calculates measurements using geometric approximations (**elliptical body model**).
5. Returns measurement data in **JSON format**.


## How to Run

```bash
pip install -r requirements.txt
python app.py
```


# API Endpoint

**POST** `/measurements`

> ℹ️ For reference, see the images placed  in the root directory.

---
##  Request
Send a `multipart/form-data` **POST** request with the following fields:

- **`front_image`**: JPEG/PNG image captured from the front *(required)*
- **`side_image`** *(optional)*: JPEG/PNG image from the side *(for better accuracy)*
- **`user_height_cm`** : Real height of the person (in cm) for more precise calibration

---

###  Example using `curl`

```bash
curl -X POST http://localhost:5000/measurements \
  -F "front_image=@front.jpg" \
  -F "side_image=@side.jpg" \
  -F "user_height_cm=170"
```

# Measurements Provided

| **Measurement Name**     | **Description**                                                   |
|--------------------------|-------------------------------------------------------------------|
| `shoulder_width`         | Distance between left and right shoulders                        |
| `chest_width`            | Width at chest level                                              |
| `chest_circumference`    | Estimated chest circumference                                     |
| `waist_width`            | Width at waist level                                              |
| `waist`                  | Estimated waist circumference                                     |
| `hip_width`              | Distance between left and right hips                             |
| `hip_circumference`      | Estimated hip circumference *(if side image is given)*           |

---

> 📌 **Note:**  
> The system uses **AI depth maps** and **contour-based width detection**.  
> Final measurements may have a **±2–3 cm variance** depending on image quality and user alignment.


# Integration in Fashion E-Commerce

This solution is plug-and-play for:

- **E-commerce brands** offering size suggestions or virtual try-ons.
- **Tailoring platforms** wanting remote client measurements.
- **Clothing manufacturers** personalizing size charts for customers.
- **Fashion mobile apps** for custom-fitted clothing suggestions.

Simply integrate this API into your frontend — mobile or web — to collect two photos and retrieve exact measurements.


## 🤝 Contributions

PRs and suggestions are welcome! Fork this repo, raise an issue, or open a pull request.

## 📜 License

MIT License. Feel free to use this for personal or commercial projects — just give credit.



