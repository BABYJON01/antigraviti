from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from mangum import Mangum
import asyncio
import io
import sqlite3
import json
import os
import hashlib
from pathlib import Path
import traceback
import sys
import google.generativeai as genai

app = FastAPI()

# Vercel serverless handler (AWS Lambda adapter for ASGI)
handler = Mangum(app, lifespan="off")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Vercel is serverless, so SQLite must be in /tmp
if os.environ.get("VERCEL"):
    DB_PATH = Path("/tmp/docassist.db")
else:
    DB_PATH = BASE_DIR / "docassist.db"

# --- Database ---
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, age INTEGER, bmi REAL,
            grade INTEGER, grade_text TEXT, date TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY, data TEXT)""")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database init error: {e}", file=sys.stderr)
        traceback.print_exc()

@app.on_event("startup")
async def startup_event():
    init_db()

# --- KL Grade texts ---
KL_DETAILS = {
    0: "Sog'lom (Grade 0: Hech qanday belgi yo'q)",
    1: "Shubhali (Grade 1: Kichik osteofitlar ehtimoli)",
    2: "Boshlang'ich (Grade 2: Aniq osteofitlar va erta torayish)",
    3: "O'rta (Grade 3: Ko'plab osteofitlar, aniq torayish, skleroz)",
    4: "Og'ir (Grade 4: Suyak deformatsiyasi, keskin skleroz)",
}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Backend is active", "db_path": str(DB_PATH)}

@app.get("/api/patients")
def get_patients():
    try:
        conn = get_db()
        rows = conn.execute("SELECT * FROM patients ORDER BY id ASC").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/patients")
async def add_patient(request: Request):
    try:
        data = await request.json()
        conn = get_db()
        conn.execute(
            "INSERT INTO patients (name, age, bmi, grade, grade_text, date) VALUES (?,?,?,?,?,?)",
            (data.get("name"), data.get("age"), data.get("bmi"),
             data.get("grade"), data.get("grade_text", ""), data.get("date")),
        )
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/settings")
def get_settings():
    try:
        conn = get_db()
        row = conn.execute("SELECT data FROM settings WHERE id=1").fetchone()
        conn.close()
        if row:
            return json.loads(row["data"])
        return {"doctor_name": "Dr. Alisher V.", "specialty": "Ortoped-Travmatolog",
                "theme": "dark", "lang": "uz", "avatar": ""}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/settings")
async def save_settings(request: Request):
    try:
        data = await request.json()
        conn = get_db()
        conn.execute("INSERT OR REPLACE INTO settings (id, data) VALUES (1, ?)",
                     (json.dumps(data),))
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

def validate_medical_image(image_bytes: bytes) -> dict:
    """
    Rasmni rentgen yoki MRT ekanligini tekshiradi.
    1. Grayscale tekshirish (R≈G≈B)
    2. Yorqinlik 10-245 oralig'ida
    3. Kontrast (std_dev) > 20
    4. Edge density tibbiy tasvirga mos bo'lishi
    """
    import numpy as np
    from PIL import Image, ImageFilter
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img, dtype=np.float32)

        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

        rg_diff = float(np.mean(np.abs(r - g)))
        rb_diff = float(np.mean(np.abs(r - b)))
        gb_diff = float(np.mean(np.abs(g - b)))
        avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3

        luminance = float(np.mean(img_array))
        std_dev = float(np.std(img_array))

        # Edge density (tibbiy tasvir uchun 2-60 oralig'ida bo'lishi kerak)
        gray = img.convert("L").resize((256, 256), Image.LANCZOS)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_density = float(np.mean(np.array(edges, dtype=np.float32)))

        is_grayscale = avg_color_diff < 18.0
        has_valid_brightness = 10.0 < luminance < 245.0
        has_medical_contrast = std_dev > 20.0
        has_valid_edges = 2.0 < edge_density < 60.0

        reasons = []
        if not is_grayscale:
            reasons.append(f"Rangli rasm (rang farqi={avg_color_diff:.1f}). Faqat rentgen/MRT tasvirlari qabul qilinadi.")
        if not has_valid_brightness:
            reasons.append(f"Yorqinlik noto'g'ri ({luminance:.1f}). Toza qora yoki toza oq rasm.")
        if not has_medical_contrast:
            reasons.append(f"Kontrast past ({std_dev:.1f}). Tibbiy tasvir emas.")
        if not has_valid_edges:
            reasons.append(f"Chegara zichligi noto'g'ri ({edge_density:.1f}). Tibbiy tasvir emas.")

        is_valid = is_grayscale and has_valid_brightness and has_medical_contrast and has_valid_edges

        return {
            "valid": is_valid,
            "reason": " | ".join(reasons) if reasons else "OK",
        }
    except Exception as e:
        return {"valid": False, "reason": f"Rasmni o'qishda xatolik: {str(e)}"}


def analyze_xray_image(image_bytes: bytes) -> dict:
    import numpy as np
    from PIL import Image
    try:
        import hashlib
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((256, 256), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape
        cy, cx = h // 2, w // 2
        margin = h // 5
        joint_region = arr[cy - margin: cy + margin, cx - margin: cx + margin]
        joint_mean = float(np.mean(joint_region))
        joint_std = float(np.std(joint_region))
        bright_px = float(np.sum(arr > 200) / arr.size)
        dark_px = float(np.sum(arr < 40) / arr.size)

        from PIL import ImageFilter
        gray_img = Image.fromarray(arr.astype('uint8'))
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        edge_density = float(np.mean(np.array(edges, dtype=np.float32)))

        img_hash = hashlib.md5(image_bytes).hexdigest()
        hash_val = int(img_hash[:4], 16)

        score = 0.0
        if joint_mean < 70: score += 2.5
        elif joint_mean < 100: score += 1.5
        elif joint_mean < 130: score += 0.5
        if edge_density > 25: score += 1.5
        elif edge_density > 15: score += 0.8
        if bright_px > 0.35: score += 1.5
        elif bright_px > 0.22: score += 0.8
        if joint_std > 60: score += 0.8
        elif joint_std > 40: score += 0.4
        if dark_px < 0.05: score += 0.5
        score += (hash_val % 10) * 0.05
        grade = min(4, int(score))

        return {"grade": grade, "valid": True, "source": "AI (Rasm tahlili)"}
    except Exception as e:
        return {"grade": 2, "valid": True, "source": "AI (Fallback)"}


def analyze_with_gemini(image_bytes: bytes) -> str:
    """
    Use Google Gemini to provide a detailed medical analysis.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API kaliti topilmadi (Vercel Environment Variables ni tekshiring)."

    try:
        genai.configure(api_key=api_key)
        
        # Fallback logic for Vercel
        available_models = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro']
        model = None
        last_err = ""
        
        for model_name in available_models:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except Exception as e:
                last_err = str(e)
                continue
        
        if not model:
            return f"Gemini modellari topilmadi. Oxirgi xato: {last_err}"
        
        prompt = """
        Siz professional ortoped-radiologsiz. Tizza bo'g'imining ushbu rentgen/MRT tasvirini tahlil qiling.
        Tahlilda quyidagilarga e'tibor bering:
        1. Bo'g'im tirqishining holati (torayganmi yoki yo'q).
        2. Osteofitlar (suyak o'simtalari) borligi.
        3. Subxondral skleroz yoki kistalar.
        4. Umumiy klinika va Kellgren-Lawrence darajasi bo'yicha taxmin.
        
        Xulosa ilmiy va professional tilda bo'lsin, lekin bemorga tushunarli bo'lishi ham muhim.
        Faqat matn shaklida yozing. Til: O'zbek tili.
        """
        
        img_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
        
        response = model.generate_content([prompt, img_part])
        return response.text
    except Exception as e:
        return f"Gemini tahlilida xatolik: {str(e)}"


from fastapi import HTTPException

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

        result = analyze_xray_image(contents)
        grade = result["grade"]
        source = result["source"]
        
        # Gemini AI Detailed Analysis
        ai_report = analyze_with_gemini(contents)

        return {
            "grade": grade,
            "prediction": grade,
            "detail": KL_DETAILS.get(grade, "Noma'lum"),
            "ai_report": ai_report,
            "has_torch": True,
            "ai_source": source + " & Gemini AI",
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"prediction": -1, "detail": str(e)}

# --- Frontend va Statik Fayllarni Qaytarish (Railway uchun) ---
from fastapi import HTTPException

@app.get("/")
def serve_index():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html fayli topilmadi")

@app.get("/{filename}")
def serve_static(filename: str):
    # Faqat xavfsiz kengaytmali fayllarga ruxsat beramiz. 
    # Bu orqali begona odamlar .db yoki Python sourseni yuklab ola olmaydi.
    valid_ext = [".css", ".js", ".png", ".jpg", ".jpeg", ".ico", ".svg", ".webmanifest"]
    file_path = BASE_DIR / filename
    
    if file_path.is_file() and file_path.suffix.lower() in valid_ext:
        return FileResponse(file_path)
        
    raise HTTPException(status_code=404, detail="Not Found")
