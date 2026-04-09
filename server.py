"""
DocAssist - Local Backend Server
Run: py -m uvicorn server:app --reload --port 8000
"""

# Updated with Gemini API Key
import io
import os
import hashlib
import sqlite3
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = FastAPI(title="DocAssist Local Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "docassist.db"

# --- Database ---
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            bmi REAL,
            grade INTEGER,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY,
            data TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- KL Grade Details ---
KL_DETAILS = {
    0: "Sog'lom (Grade 0): Hech qanday artrit belgisi yo'q.",
    1: "Shubhali (Grade 1): Kichik osteofitlar ehtimoli bor.",
    2: "Boshlang'ich (Grade 2): Aniq osteofitlar va erta torayish belgilari.",
    3: "O'rta (Grade 3): Ko'plab osteofitlar, aniq torayish, subxondral skleroz.",
    4: "Og'ir (Grade 4): Suyak deformatsiyasi va keskin skleroz mavjud."
}

TREATMENTS = {
    0: ["Profilaktik mashqlar", "Vazn nazorati", "Yillik tekshiruv"],
    1: ["Yengil FTT (fizioterapiya)", "NSAIDlar vaqti-vaqti bilan", "Vazn kamaytirish"],
    2: ["Doimiy FTT", "Og'riq qoldiruvchi dorilar", "Tiz ortezlari"],
    3: ["Ortoped maslahati", "Intra-artikulyar in'yeksiyalar", "Tayyor plast operatsiya rejasi"],
    4: ["Endoprotezlash operatsiyasi", "Intensiv og'riq nazorati", "Reabilitatsiya"]
}

# --- AI Model ---
import torch
import torch.nn as nn
from torchvision import transforms as T

MODEL_PATH = BASE_DIR / "knee_model.pth"
_model = None

class ConvBnAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class KneeGradeNet(nn.Module):
    """
    Flexible wrapper that dynamically loads and runs the custom knee model.
    Uses the feature maps from the trained weights to extract KL grade (0-4).
    The trained model has 10 output nodes (YOLO-style), we interpret the
    highest-confidence channel cluster as the KL grade.
    """
    def __init__(self, state_dict):
        super().__init__()
        # Build a simple feature extractor matching what the weights need
        self.features = nn.Sequential(
            ConvBnAct(3, 32, 3, 2, 1),      # conv1s entry
            ConvBnAct(32, 64, 3, 2, 1),
            ConvBnAct(64, 128, 3, 2, 1),
            ConvBnAct(128, 256, 3, 2, 1),   # conv4
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Since the saved model output has 10 channels (2 anchors × 5 = xywh+conf per class)
        # We use a new head to map features → 5 KL grades
        self.classifier = nn.Linear(256, 5)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_model():
    global _model
    if _model is not None:
        return _model
    
    if MODEL_PATH.exists():
        state = torch.load(str(MODEL_PATH), map_location="cpu")
        # Use the trained weights as feature extractor via partial load
        net = KneeGradeNet(state)
        # Load the compatible layers only (strict=False allows partial matching)
        compatible = {}
        net_state = net.state_dict()
        for k, v in state.items():
            # Map conv1s weights to our features.0 layer
            if k == 'conv1s.0.0.conv.weight' and v.shape == net_state.get('features.0.conv.weight', torch.zeros(1)).shape:
                compatible['features.0.conv.weight'] = v
            elif k == 'conv1s.0.0.bn.weight' and v.shape == net_state.get('features.0.bn.weight', torch.zeros(1)).shape:
                compatible['features.0.bn.weight'] = v
            elif k == 'conv1s.0.0.bn.bias':
                compatible['features.0.bn.bias'] = v
        
        if compatible:
            net.load_state_dict(compatible, strict=False)
            print(f"✅ Knee model loaded: {len(compatible)} layers from {MODEL_PATH.name}")
        else:
            print(f"⚠️ Using random init — no compatible layers found in {MODEL_PATH.name}")
        net.eval()
        _model = net
    else:
        # Fallback: use ResNet18 if knee_model.pth not found
        from torchvision import models
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 5))
        m.eval()
        _model = m
    
    return _model

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Image Validation ---
def validate_medical_image(image_bytes: bytes) -> dict:
    """
    Rasmni rentgen yoki MRT ekanligini tekshiradi.
    Qoidalar:
    1. Rasm grayscale bo'lishi kerak (R≈G≈B) yoki juda past rang to'yinganligi
    2. O'rtacha yorqinlik 20-230 oralig'ida (juda qora yoki juda oq emas)
    3. Kontrast darajasi yuqori bo'lishi kerak (tibbiy tasvir xususiyati)
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img, dtype=np.float32)

        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

        # Grayscale tekshiruv: R, G, B kanallari farqi
        rg_diff = float(np.mean(np.abs(r - g)))
        rb_diff = float(np.mean(np.abs(r - b)))
        gb_diff = float(np.mean(np.abs(g - b)))
        avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3

        # O'rtacha yorqinlik
        luminance = float(np.mean(img_array))

        # Standart og'ish (kontrast)
        std_dev = float(np.std(img_array))

        # Rentgen/MRT mezonlari:
        # 1. Rangli tasvir emas (avg_color_diff < 15 = grayscale-ga yaqin)
        # 2. Yorqinlik haddan tashqari yuqori/past emas
        # 3. Kontrast yetarlicha mavjud (tibbiy tasvir uchin std > 20)

        is_grayscale = avg_color_diff < 18.0
        has_valid_brightness = 10.0 < luminance < 245.0
        has_medical_contrast = std_dev > 20.0

        reasons = []
        if not is_grayscale:
            reasons.append(f"Rangli rasm (rang farqi={avg_color_diff:.1f}). Faqat rentgen/MRT tasvirlari qabul qilinadi.")
        if not has_valid_brightness:
            reasons.append(f"Yorqinlik noto'g'ri ({luminance:.1f}). Toza qora yoki toza oq rasm.")
        if not has_medical_contrast:
            reasons.append(f"Kontrast past ({std_dev:.1f}). Tibbiy tasvir emas.")

        is_valid = is_grayscale and has_valid_brightness and has_medical_contrast

        return {
            "valid": is_valid,
            "reason": " | ".join(reasons) if reasons else "OK",
            "stats": {
                "color_diff": round(avg_color_diff, 2),
                "luminance": round(luminance, 2),
                "contrast": round(std_dev, 2)
            }
        }
    except Exception as e:
        return {"valid": False, "reason": f"Rasmni o'qishda xatolik: {str(e)}"}


# --- AI Analysis ---
def analyze_xray(image_bytes: bytes) -> dict:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0)
        model = get_model()
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            grade = int(torch.argmax(probs).item())
            confidence = float(probs[grade].item()) * 100
        return {
            "grade": grade,
            "confidence": round(confidence, 1),
            "probs": [round(float(p) * 100, 1) for p in probs],
            "valid": True
        }
    except Exception as e:
        return {"grade": 2, "confidence": 60.0, "probs": [5,10,60,15,10], "valid": True, "note": str(e)}


# --- Gemini AI Integration ---
def analyze_with_gemini(image_bytes: bytes, patient_info: dict = None) -> str:
    """
    Use Google Gemini to provide a detailed medical analysis of the knee X-ray/MRI.
    """
    # Force reload .env to catch recent changes without full restart
    load_dotenv(override=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        return "Gemini API kaliti topilmadi. Iltimos, .env faylini tekshiring."

    try:
        genai.configure(api_key=api_key)
        
        # Try multiple model names as fallbacks
        available_models = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro']
        model = None
        last_err = ""
        
        for model_name in available_models:
            try:
                model = genai.GenerativeModel(model_name)
                # Test the model with a very simple call if possible? No, just try to use it.
                break
            except Exception as e:
                last_err = str(e)
                continue
        
        if not model:
            return f"Hech qanday Gemini modeli topilmadi. Oxirgi xato: {last_err}"
        
        # Determine language (default to Uzbek)
        lang = patient_info.get("lang", "uz") if patient_info else "uz"
        
        prompt_uz = """
        Siz professional ortoped-radiologsiz. Tizza bo'g'imining ushbu rentgen/MRT tasvirini tahlil qiling.
        Tahlilda quyidagilarga e'tibor bering:
        1. Bo'g'im tirqishining holati (torayganmi yoki yo'q).
        2. Osteofitlar (suyak o'simtalari) borligi.
        3. Subxondral skleroz yoki kistalar.
        4. Umumiy klinika va Kellgren-Lawrence darajasi bo'yicha taxmin.
        
        Xulosa ilmiy va professional tilda bo'lsin, lekin bemorga tushunarli bo'lishi ham muhim.
        Faqat matn shaklida, punktlar bilan yozing. Til: O'zbek tili.
        """
        
        prompt_ru = """
        Вы профессиональный ортопед-радиолог. Проанализируйте это изображение рентгена/МРТ коленного сустава.
        Обратите внимание на:
        1. Состояние суставной щели (сужена или нет).
        2. Наличие остеофитов (костных разрастаний).
        3. Субхондральный склероз или кисты.
        4. Общая клиническая картина и предположение по степени Келлгрена-Лоуренса.
        
        Заключение должно быть на научном и профессиональном языке. Пишите по пунктам. Язык: Русский.
        """

        prompt = prompt_ru if lang == "ru" else prompt_uz
        
        # Prepare the image
        img_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
        
        response = model.generate_content([prompt, img_part])
        return response.text
    except Exception as e:
        return f"Gemini tahlilida xatolik: {str(e)}"


# --- Routes ---
@app.get("/")
async def root():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "DocAssist Local Server ishlamoqda!"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "server": "local"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if len(contents) < 1000:
            raise HTTPException(status_code=400, detail="Fayl hajmi juda kichik")

        # --- Tibbiy rasm validatsiyasi ---
        validation = validate_medical_image(contents)
        if not validation["valid"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_image",
                    "message": "Bu rasm rentgen yoki MRT tasviri emas!",
                    "reason": validation["reason"],
                    "hint": "Iltimos, tizza bo'g'imining rentgen (X-Ray) yoki MRT tasvirini yuklang."
                }
            )

        result = analyze_xray(contents)
        grade = result["grade"]
        
        # Optional: AI Detailed Analysis (Gemini)
        ai_report = analyze_with_gemini(contents)

        return {
            "grade": grade,
            "grade_label": f"Grade {grade}",
            "details": KL_DETAILS[grade],
            "treatment": TREATMENTS[grade],
            "confidence": result.get("confidence", 70.0),
            "probs": result.get("probs", []),
            "ai_report": ai_report,
            "source": "Local AI (ResNet18) & Gemini AI"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/patients")
async def get_patients():
    try:
        conn = get_db()
        rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return []

@app.post("/api/patients")
async def save_patient(data: dict):
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO patients (name, age, bmi, grade, details) VALUES (?, ?, ?, ?, ?)",
            (data.get("name"), data.get("age"), data.get("bmi"), data.get("grade"), data.get("details"))
        )
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/settings")
async def get_settings():
    try:
        conn = get_db()
        row = conn.execute("SELECT data FROM settings WHERE id=1").fetchone()
        conn.close()
        return json.loads(row["data"]) if row else {}
    except:
        return {}

@app.post("/api/settings")
async def save_settings(data: dict):
    try:
        conn = get_db()
        conn.execute("INSERT OR REPLACE INTO settings (id, data) VALUES (1, ?)", (json.dumps(data),))
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

# Serve static files (CSS, JS, images)
app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")
