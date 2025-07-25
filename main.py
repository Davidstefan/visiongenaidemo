import cv2
import time
import threading
import numpy as np
import base64
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import json
from openai import AzureOpenAI
import re

# === Load Azure config from config.json ===
with open("config.json", "r") as f:
    config = json.load(f)

client = AzureOpenAI(
    api_key=config["api_key"],
    api_version=config["api_version"],
    azure_endpoint=config["endpoint"],
)

# === Load YOLO model
model = YOLO("yolov8n.pt")

# Global states
latest_description = "Even geduld..."
latest_emotie = ""
latest_sfeer = ""
latest_positief = ""
latest_negatief = ""
last_frame = None
last_detection_summary = ""
gpt_snapshot_frame = None  # Frame dat naar GPT wordt gestuurd


# === Camera selectie functies ===
def list_available_cameras(max_tested=5):
    """Detecteer beschikbare camera's (indices 0..max_tested)."""
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


# === JSON cleanup helper ===
def clean_llm_json_response(content):
    try:
        json_str = re.search(r"\{.*\}", content, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception:
        return None


# === LLM-query ===
def query_azure_openai(img_b64):
    prompt_text = (
        "Beschrijf wat te zien is in exact 20 woorden (kort en nauwkeurig). "
        "Geef daarna trefwoorden voor EMOTIE, SFEER, POSITIEF ELEMENT en NEGATIEF ELEMENT. "
        "Geef output ALLEEN als JSON, bijvoorbeeld:\n"
        "{ \"beschrijving\": \"<20 woorden>\", \"emotie\": \"<woord>\", \"sfeer\": \"<woord>\", "
        "\"positief\": \"<woord>\", \"negatief\": \"<woord>\" }"
    )

    try:
        response = client.chat.completions.create(
            model=config["deployment"],
            max_tokens=200,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": "Je bent een visuele assistent die scènes nauwkeurig beschrijft. "
                               "Geef ALLEEN JSON terug, geen extra tekst.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                },
            ],
        )
        content = response.choices[0].message.content.strip()
        parsed = clean_llm_json_response(content)
        if parsed:
            return (
                f"{config['deployment']}: {parsed.get('beschrijving', '')}",
                parsed.get("emotie", ""),
                parsed.get("sfeer", ""),
                parsed.get("positief", ""),
                parsed.get("negatief", ""),
            )
        else:
            return f"{config['deployment']}: {content}", "", "", "", ""
    except Exception as e:
        return f"[Fout bij Azure GPT-4o]: {e}", "", "", "", ""


# === LLM thread ===
def update_description_every_10s():
    global latest_description, latest_emotie, latest_sfeer, latest_positief, latest_negatief
    global last_frame, gpt_snapshot_frame

    while True:
        if last_frame is not None:
            try:
                gpt_snapshot_frame = last_frame.copy()

                small_frame = cv2.resize(gpt_snapshot_frame, (0, 0), fx=0.5, fy=0.5)
                img_pil = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
                buffer = BytesIO()
                img_pil.save(buffer, format="JPEG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                beschrijving, emotie, sfeer, positief, negatief = query_azure_openai(img_b64)
                latest_description = beschrijving
                latest_emotie = emotie
                latest_sfeer = sfeer
                latest_positief = positief
                latest_negatief = negatief
            except Exception as e:
                latest_description = f"[Error]: {e}"
        time.sleep(10)


# === Tekst wrapping helper ===
def wrap_text(text, font, scale, thickness, max_width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = current + (" " if current else "") + word
        size = cv2.getTextSize(test, font, scale, thickness)[0]
        if size[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


# === Webcam selectie ===
available_cams = list_available_cameras(5)
if not available_cams:
    raise RuntimeError("No webcam found.")
print("Available:", available_cams)

selected_cam = int(input(f"Select webcam (standard {available_cams[0]}): ") or available_cams[0])
cap = cv2.VideoCapture(selected_cam)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open {selected_cam}.")

# === Window setup ===
window_name = "AI Vision Demo (C/F = stop/fullscreen)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
fullscreen = False

threading.Thread(target=update_description_every_10s, daemon=True).start()

# === Main loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    last_frame = frame.copy()

    results = model(frame, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    names = results.names

    # Kleuren per object
    object_colors = {}
    np.random.seed(42)
    for cls_id in np.unique(class_ids):
        object_colors[int(cls_id)] = tuple(int(x) for x in np.random.randint(0, 255, 3))

    for box, conf, cls in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = names[int(cls)]
        color = object_colors.get(int(cls), (0, 160, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Overlay GPT description
    font_scale = 0.5
    line_height = 22
    max_width = int(frame.shape[1] * 0.45)
    margin, padding = 20, 14
    text_color = (0, 255, 180)
    bg_color = (30, 30, 30)
    alpha = 0.75

    desc_lines = wrap_text(latest_description, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2, max_width)
    extra_lines = [
        f"EMOTIE: {latest_emotie}",
        f"SFEER: {latest_sfeer}",
        f"POSITIEF: {latest_positief}",
        f"NEGATIEF: {latest_negatief}",
    ]
    all_lines = desc_lines + extra_lines

    box_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0] for l in all_lines) + 2 * padding
    box_h = line_height * len(all_lines) + 2 * padding
    x1, y1 = frame.shape[1] - box_w - margin, margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, line in enumerate(all_lines):
        ty = y1 + padding + (i + 1) * line_height - 5
        cv2.putText(frame, line, (x1 + padding, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, 2)

    # Thumbnail GPT snapshot
    if gpt_snapshot_frame is not None:
        thumb = cv2.resize(gpt_snapshot_frame, (150, 150))
        thumb_x, thumb_y = margin, margin
        frame[thumb_y:thumb_y + 150, thumb_x:thumb_x + 150] = thumb
        cv2.rectangle(frame, (thumb_x, thumb_y), (thumb_x + 150, thumb_y + 150), (255, 255, 255), 2)
        cv2.putText(frame, "Snapshot", (thumb_x + 5, thumb_y + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key in [ord("q"), ord("c")]:
        print("⛔ Gestopt door gebruiker.")
        break
    elif key == ord("f"):
        fullscreen = not fullscreen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
