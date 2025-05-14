<p align="center">
  <strong>🏆 This project was awarded Most Technical at FullyHacks 2025 🏆</strong>
  <br>
   <div><img alt="ASTRO ASL logo" src="https://github.com/user-attachments/assets/e89141a7-9766-471d-bcea-59eec2efe7cd"></div>
</p>

# 🪐 FullyHacks 2025
[FullyHacks](https://fullyhacks.acmcsuf.com/) is the largest 24-hour Hackathon at CSUF hosted by [ACMCSUF](https://acmcsuf.com/) 
<div><img alt="FullyHacks logo" src="https://github.com/user-attachments/assets/5fbc4f4b-071a-4a9b-95af-d8bef0ceab10" width="200"></div>

# 👽 ASTRO ASL
No sound in space? No Problem. Real-time ASL recognition via deep learning 💫

## 📝 Summary  
**ASTRO ASL (American Sign Language)** is an AI-powered sign language transcriber designed specifically for space missions. By capturing and interpreting ASL gestures in real time, ASTRO ASL bridges communication gaps in environments where traditional audio communication fails.

## 🎯 Mission Statement  
Our mission is to empower astronauts with a reliable, silent communication tool.  
We strive to:  
- Enable non-verbal communication in space where sound doesn't travel  
- Support inclusivity for deaf or hard-of-hearing astronauts  
- Enhance safety through clear, real-time gesture recognition  

By combining cutting-edge AI with intuitive design, ASTRO ASL provides a seamless way for astronauts to interact without relying on sound — ensuring mission-critical information is never lost in translation.

## ⚠️ The Problem  
Sound doesn’t travel in the vacuum of space, making traditional spoken communication unreliable during spacewalks or in loud environments.  
- Verbal communication is limited in spacewalks  
- Audio equipment can malfunction or be blocked by suits  
- Safety risks increase when commands aren't clearly heard  

> **ASTRO ASL provides a hands-on solution to a soundless environment.**


## ✨ Features  

| Feature           | Description                                                  |
|-------------------|--------------------------------------------------------------|
| 🤖 AI Recognition | Real-time ASL interpretation via onboard camera systems      |
| 🧠 Onboard ML     | No internet needed — edge processing for zero-latency use    |
| 📊 Scribe Logs   | Automatically logs conversations for mission review          |

## 🌟 How Are We Unique?  

- **Tech Used**: TensorFlow, OpenCV, Mediapipe, Scikit-learn
- **Built for Space**: Engineered to work in zero-gravity and vacuum conditions  
- **Offline AI**: Works without any internet connection, runs locally  
- **Mission-Critical UX**: Simplified UI for high-pressure scenarios 

# 🚀 Steps to Use

## 1️⃣ Clone Repository
Ensure you have Git installed on your system. Then, clone the repository and navigate into the project folder:

```bash
git clone https://github.com/isliese/astro-asl.git
cd astro-asl
```

## 2️⃣ Set Up Environment
Create and activate a virtual environment to manage dependencies.

### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

## 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 4️⃣ Train the Model

```bash
python training_model.py
```

⚠️ **Note:** Custom datasets may require modifications to the training script.

## 5️⃣ Run the Application
Launch the web UI and transcriber:

```bash
python app.py
```

## 🔍 Troubleshooting
| Issue | Solution |
|-------|----------|
| Performance Slowdowns | Close programs or increase memory allocation |
| False Positives/Negatives | Remove neural net layers to reduce overfitting |
| Installation Issues | Verify Python version is 3.10-3.11 and install dependencies |

### 🚧 Limitations
- **Model Efficiency** - Model is not future proofed and may be overreliant on dataset.
- **Hand Isolation** - To have a clean image of the gesture, the hand must be isolated.
- **Lighting Requirement** - To be recognized by the model, the hand must be in good lighting as well.


## 🤝 Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Submit a pull request with a detailed description of your changes

## 👥 Team
| Role | Member |
|------|--------|
| Backend / Research | Owin |
| Backend / Research | Jay |
| Frontend / Design | Isla |
| Frontend / Design | Sema |

## 🙏 Credits  

This project uses the following technologies, libraries, and datasets:

### Languages:
- **Python**
- **HTML**
- **CSS**
- **JS**

### Libraries:
- **Flask** – A micro web framework for Python.
- **TensorFlow** – An open-source library for numerical computation and large-scale machine learning.
- **Scikit-learn** – A machine learning library, used for building the Random Forest model.
- **NumPy** – A library for numerical computing in Python.
- **MediaPipe** – A library for real-time computer vision.
### Dataset:
[American Sign Language Dataset](https://www.kaggle.com/datasets/ayuraj/american-sign-language-dataset) – A dataset used for training the the model.

### Explore More:
 [Astro ASL Slideshow](https://docs.google.com/presentation/d/12n0f3zPviuIEXO6e-eKsDheuZ6P-uuRe7ssC2XmmU9Y/edit?usp=sharing) <br>
 [Devpost Project Link](https://devpost.com/software/astro-asl?ref_content=my-projects-tab&ref_feature=my_projects)

## 📄 License
© 2025 | AstroASL Team - Fullyhacks @ CSUF
