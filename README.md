# 📷 AI Vision Demo – GPT-4o + YOLOv8 + Webcam

This demo combines **YOLOv8 object detection** with **Azure OpenAI's GPT-4o (vision model)** to generate real-time natural language descriptions of what your webcam sees.

---

## 📁 Files

- `main.py` – main script that runs YOLOv8 detection and GPT-4o vision
- `requirements.txt` – all required Python packages
- `config.json` – holds your personal Azure API credentials (you must create this)

---

## ⚙️ Requirements

- Python 3.8 or higher
- A webcam
- An Azure OpenAI resource with a deployed GPT-4o model

---

## 🔧 Setup Instructions

1. Clone the repo or place the files in a folder

2. (Optional) Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. Install required Python packages
   ```bash
   pip install -r requirements.txt
   ```

4. Download YOLOv8n model manually  
   Download the small YOLOv8n model here:  
   🔗 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt  
   Place the file `yolov8n.pt` in the same directory as `main.py`.

---

## 🔐 Create `config.json`

Create a file called `config.json` in the same folder as `main.py`:

```json
{
  "api_key": "<YOUR_AZURE_OPENAI_API_KEY>",
  "endpoint": "https://<your-resource>.openai.azure.com/",
  "deployment": "gpt-4o",
  "api_version": "2024-12-01-preview"
}
```

> ⚠️ **Never commit this file to public repositories.**

---

## ▶️ Run the demo

Open a terminal in the folder and run:

```bash
python main.py
```

If `python` is not recognized, run it like this:

```powershell
& "C:\Path\To\Python.exe" main.py
```

---

## ✅ Features

- Live webcam feed with object detection via YOLOv8
- Every 10 seconds:
  - A frame is captured and sent to GPT-4o as a base64 image
  - A summary of detected objects is included in the prompt
  - GPT-4o returns a two-sentence impression in Dutch
- The output is displayed in the top-right corner of the video
- Press `Q` or `C` to stop the demo

---

## 🚧 Known limitations

- Facial expressions may only be recognized if the face is large and well-lit
- GPT-4o does not always infer emotional states unless prompted clearly
- Background is intentionally ignored in the prompt unless something unusual occurs

---

## 📜 License

This code is for demonstration and educational purposes.  
© 2025 David Stefan. Use at your own risk.
