from dotenv import load_dotenv
load_dotenv()

import os, base64, random, tempfile, concurrent.futures
import gradio as gr
import numpy as np
import requests
import fal_client
from PIL import Image, ImageEnhance, ImageFilter
import io

MAX_SEED = 999999

# ─────────────────────────────────────────────────────────────────────────────
# NEW MODEL: fal-ai/image-apps-v2/virtual-try-on
#
#   Much better than IDM-VTON:
#     • Natively supports preserve_pose
#     • 4K output quality
#     • Aspect ratio control (3:4 default — perfect for fashion)
#     • Cleaner draping, better fabric detail
#
#   We generate 5 looks by varying aspect_ratio and preserve_pose.
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_CONFIGS = [
    {"label": "✦ Look 1 – Standard",      "preserve_pose": True,  "aspect_ratio": "3:4",  "seed_offset": 0},
    {"label": "✦ Look 2 – Free Flow",     "preserve_pose": False, "aspect_ratio": "3:4",  "seed_offset": 1},
    {"label": "✦ Look 3 – Portrait",      "preserve_pose": True,  "aspect_ratio": "9:16", "seed_offset": 2},
    {"label": "✦ Look 4 – Square Crop",   "preserve_pose": True,  "aspect_ratio": "1:1",  "seed_offset": 3},
    {"label": "✦ Look 5 – Reposed",       "preserve_pose": False, "aspect_ratio": "9:16", "seed_offset": 4},
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def pad_white(pil_img, w=768, h=1024):
    pil_img = pil_img.convert("RGB")
    scale   = min(w / pil_img.width, h / pil_img.height)
    nw, nh  = int(pil_img.width * scale), int(pil_img.height * scale)
    r       = pil_img.resize((nw, nh), Image.LANCZOS)
    c       = Image.new("RGB", (w, h), (255, 255, 255))
    c.paste(r, ((w - nw) // 2, (h - nh) // 2))
    return c


def light_enhance(pil_img):
    img = pil_img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=85, threshold=3))
    img = ImageEnhance.Color(img).enhance(1.10)
    img = ImageEnhance.Contrast(img).enhance(1.06)
    return img


def upload_pil(pil_img):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        pil_img.save(f.name, "JPEG", quality=97)
        path = f.name
    try:
        return fal_client.upload_file(path)
    finally:
        try: os.remove(path)
        except: pass


def download_pil(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def extract_url(result):
    if not result: return None
    if "image" in result:
        v = result["image"]
        return v.get("url") if isinstance(v, dict) else v
    if "images" in result and result["images"]:
        v = result["images"][0]
        return v.get("url") if isinstance(v, dict) else v
    return None


def remove_bg(pil_img):
    try:
        url     = upload_pil(pil_img)
        res     = fal_client.subscribe("fal-ai/imageutils/rembg",
                                       arguments={"image_url": url})
        img_url = extract_url(res)
        if img_url:
            fg    = download_pil(img_url).convert("RGBA")
            white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
            white.paste(fg, mask=fg.split()[3])
            return white.convert("RGB")
    except Exception as e:
        print(f"rembg failed (non-fatal): {e}")
    return pil_img.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DESCRIBE — GPT-4o vision reads garment pieces
# ─────────────────────────────────────────────────────────────────────────────

def auto_describe(pieces):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "Pakistani traditional embroidered formal suit"

    labels  = {"front": "Kameez Front", "back": "Kameez Back",
                "shalwar": "Shalwar / Trouser", "dupatta": "Dupatta"}

    content = []
    for key, pil in pieces.items():
        b64 = pil_to_b64(pil.resize((512, 512), Image.LANCZOS))
        content.append({"type": "text", "text": f"[{labels.get(key, key)}]"})
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
        })

    content.append({
        "type": "text",
        "text": (
            "You are a Pakistani fashion expert. "
            "Describe this traditional suit in ONE concise sentence (max 40 words) "
            "focusing on: color, fabric, embroidery style, garment pieces. "
            "Use Pakistani fashion terms (kameez, shalwar, dupatta, zari, gota, etc.). "
            "Output ONLY the description sentence, nothing else."
        )
    })

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": "gpt-4o", "max_tokens": 120,
                  "messages": [{"role": "user", "content": content}]},
            timeout=30,
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Auto-describe failed (non-fatal): {e}")

    return "Pakistani traditional embroidered formal suit"


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GARMENT COMPOSITE
# ─────────────────────────────────────────────────────────────────────────────

def build_garment(pieces):
    """
    Composite all garment pieces into one clean reference image.
    Front alone → full 768×1024.
    2 pieces    → side by side.
    3-4 pieces  → 2×2 grid.
    Each piece gets its background removed first.
    """
    cleaned = []
    for pil in pieces.values():
        c = remove_bg(pil)
        cleaned.append(pad_white(c, 512, 682))

    if len(cleaned) == 1:
        return pad_white(cleaned[0], 768, 1024)

    if len(cleaned) == 2:
        canvas = Image.new("RGB", (1024, 682), (255, 255, 255))
        canvas.paste(cleaned[0], (0,   0))
        canvas.paste(cleaned[1], (512, 0))
        return pad_white(canvas, 768, 1024)

    # 3 or 4 pieces — 2×2 grid
    cols, rows = 2, (len(cleaned) + 1) // 2
    canvas = Image.new("RGB", (512 * cols, 682 * rows), (255, 255, 255))
    for i, p in enumerate(cleaned):
        canvas.paste(p, ((i % cols) * 512, (i // cols) * 682))
    return pad_white(canvas, 768, 1024)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TRY-ON CALL — uses new image-apps-v2/virtual-try-on model
# ─────────────────────────────────────────────────────────────────────────────

def run_tryon(person_url, garment_url, cfg):
    """
    Calls fal-ai/image-apps-v2/virtual-try-on — the new high-quality model.
    Parameters:
      - person_image_url
      - clothing_image_url
      - preserve_pose  (bool)
      - aspect_ratio   (e.g. "3:4", "9:16", "1:1")
    """
    result = fal_client.subscribe(
        "fal-ai/image-apps-v2/virtual-try-on",
        arguments={
            "person_image_url":    person_url,
            "clothing_image_url":  garment_url,
            "preserve_pose":       cfg["preserve_pose"],
            "aspect_ratio":        {"ratio": cfg["aspect_ratio"]},
        }
    )
    url = extract_url(result)
    if not url:
        raise RuntimeError(f"No result from image-apps-v2/virtual-try-on: {result}")
    return np.array(light_enhance(download_pil(url)))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    person_img,
    front_img, back_img, shalwar_img, dupatta_img,
    seed, randomize_seed,
    progress=gr.Progress(track_tqdm=True),
):
    if person_img is None:
        gr.Warning("Please upload a model/person photo.")
        return [None]*5 + [seed, "❌ Person photo is required"]
    if front_img is None:
        gr.Warning("Please upload at least the Kameez Front image.")
        return [None]*5 + [seed, "❌ Kameez Front image is required"]

    fal_key = os.environ.get("FAL_KEY", "").strip()
    if not fal_key or ":" not in fal_key:
        return [None]*5 + [seed, "❌ FAL_KEY missing/invalid (format: key_id:key_secret)"]
    os.environ["FAL_KEY"] = fal_key

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    try:
        # 1. Collect pieces
        raw    = {"front": front_img, "back": back_img,
                  "shalwar": shalwar_img, "dupatta": dupatta_img}
        pieces = {k: Image.fromarray(v).convert("RGB")
                  for k, v in raw.items() if v is not None}

        # 2. Auto-describe via GPT-4o
        progress(0.05, desc="🔍 AI reading your dress design...")
        description = auto_describe(pieces)
        print(f"Detected: {description}")

        # 3. Preprocess person
        progress(0.12, desc="👤 Preprocessing person photo...")
        person_pil = pad_white(Image.fromarray(person_img), 768, 1024)

        # 4. Build + clean garment composite
        progress(0.18, desc="✂️ Removing backgrounds & compositing garment...")
        garment_pil = build_garment(pieces)

        # 5. Upload both
        progress(0.30, desc="☁️ Uploading to fal.ai...")
        person_url  = upload_pil(person_pil)
        garment_url = upload_pil(garment_pil)

        # 6. Run 5 calls in parallel with the NEW model
        progress(0.38, desc="🎨 Generating 5 looks in parallel...")
        outputs = [None] * 5
        errors  = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(run_tryon, person_url, garment_url, OUTPUT_CONFIGS[i]): i
                for i in range(5)
            }
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]; done += 1
                progress(0.38 + (done / 5) * 0.58,
                         desc=f"✅ {OUTPUT_CONFIGS[i]['label']} done  ({done}/5)")
                try:
                    outputs[i] = fut.result()
                except Exception as e:
                    errors[i] = str(e)
                    print(f"[Look {i+1}] FAILED: {e}")

        if errors:
            msg = f"⚠️ {len(errors)} look(s) failed. Detected: {description}"
        else:
            msg = f"✅ Done!  Detected: {description}"

        return outputs + [seed, msg]

    except Exception as err:
        msg = f"❌ {err}"
        print(msg); gr.Warning(msg)
        return [None]*5 + [seed, msg]


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

css = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
:root{
    --gold:#C9A84C;--gold2:#E8C97A;--deep:#0D0618;--card:#1B0D34;
    --border:rgba(201,168,76,0.20);--text:#EDE5D8;--muted:#8A7DA0;--maroon:#7A1030;
}
*,*::before,*::after{box-sizing:border-box;}
body,.gradio-container{
    background:var(--deep)!important;
    font-family:'DM Sans',sans-serif!important;
    color:var(--text)!important;
}
.gradio-container{max-width:1400px!important;margin:0 auto!important;padding:0!important;}

#pk-header{
    text-align:center;padding:48px 24px 30px;
    border-bottom:1px solid var(--border);margin-bottom:36px;
    background:linear-gradient(180deg,rgba(122,16,48,0.18)0%,transparent 100%);
}
#pk-header h1{
    font-family:'Cormorant Garamond',serif;
    font-size:clamp(2rem,4vw,3.2rem);font-weight:700;
    color:var(--gold);letter-spacing:.05em;margin:0 0 10px;
    text-shadow:0 2px 32px rgba(201,168,76,.28);
}
#pk-header p{color:var(--muted);font-size:.8rem;letter-spacing:.16em;text-transform:uppercase;margin:0;}

.sec-head{
    font-family:'Cormorant Garamond',serif;font-size:.95rem;font-weight:600;
    color:var(--gold);letter-spacing:.09em;text-transform:uppercase;
    padding-bottom:8px;border-bottom:1px solid var(--border);margin-bottom:16px;
}

label>span{
    color:var(--gold)!important;font-size:.76rem!important;
    letter-spacing:.07em!important;text-transform:uppercase!important;font-weight:500!important;
}
input[type=text],input[type=number],textarea,select{
    background:rgba(13,6,24,.75)!important;border:1px solid var(--border)!important;
    color:var(--text)!important;border-radius:8px!important;
    font-family:'DM Sans',sans-serif!important;
}
.gr-image{
    border:2px dashed rgba(201,168,76,.22)!important;
    border-radius:12px!important;background:rgba(13,6,24,.5)!important;
    transition:border-color .2s!important;
}
.gr-image:hover{border-color:rgba(201,168,76,.5)!important;}

.tip-box{
    background:rgba(201,168,76,.06);border:1px solid rgba(201,168,76,.16);
    border-radius:8px;padding:10px 14px;font-size:.77rem;
    color:var(--muted);line-height:1.7;margin-top:10px;
}
.tip-box strong{color:var(--gold2);}

.model-badge{
    display:inline-block;background:rgba(122,16,48,0.35);
    border:1px solid var(--gold);border-radius:20px;
    padding:4px 14px;font-size:.72rem;letter-spacing:.1em;
    color:var(--gold2);text-transform:uppercase;margin-top:8px;
}

#run-btn{
    background:linear-gradient(135deg,#7A1030 0%,#9E1A3C 100%)!important;
    border:1px solid var(--gold)!important;color:var(--gold)!important;
    font-family:'Cormorant Garamond',serif!important;font-size:1.3rem!important;
    font-weight:700!important;letter-spacing:.14em!important;
    padding:22px 32px!important;border-radius:10px!important;
    width:100%!important;transition:box-shadow .25s,transform .2s!important;
    margin-top:8px!important;
}
#run-btn:hover{box-shadow:0 0 40px rgba(201,168,76,.45)!important;transform:translateY(-2px)!important;}

.result-row .gr-image{border-radius:12px!important;border:1px solid var(--border)!important;}
.gold-div{border:none;border-top:1px solid var(--border);margin:24px 0;}
#status-box textarea{font-size:.82rem!important;color:#9EE89E!important;}
#pk-footer{text-align:center;padding:24px 0 30px;color:#2E1A4A;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;}
input[type=checkbox]{accent-color:var(--gold)!important;}
"""

example_path = os.path.join(os.path.dirname(__file__), "assets")
try:    human_ex = [os.path.join(example_path,"human",h) for h in os.listdir(os.path.join(example_path,"human"))]
except: human_ex = []


with gr.Blocks(css=css, title="Pakistani Dress Virtual Try-On") as Tryon:

    gr.HTML("""
    <div id="pk-header">
        <h1>✦ Pakistani Dress Virtual Try-On ✦</h1>
        <p>Upload your suit pieces — AI does the rest</p>
        <div class="model-badge">Powered by image-apps-v2/virtual-try-on · 4K Quality</div>
    </div>""")

    # ── Row 1: Person + Garment pieces ───────────────────────────────────────
    with gr.Row(equal_height=False):

        # Person photo — left
        with gr.Column(scale=1, min_width=220):
            gr.HTML('<div class="sec-head">👤 Model Photo</div>')
            person_img = gr.Image(
                label="Person / Model — Required",
                sources="upload", type="numpy", height=400,
            )
            gr.HTML("""<div class="tip-box">
                <strong>Tips for best results:</strong><br>
                • Full-body, front-facing<br>
                • Plain / white background<br>
                • Person in plain fitted clothes<br>
                • Good even lighting
            </div>""")
            if human_ex:
                gr.Examples(inputs=person_img, examples_per_page=4,
                            examples=human_ex, label="Example models")

        # Garment pieces — right (2×2)
        with gr.Column(scale=3, min_width=600):
            gr.HTML('<div class="sec-head">👗 Dress Pieces</div>')
            with gr.Row():
                front_img   = gr.Image(label="Kameez Front ✦ Required",
                                       sources="upload", type="numpy", height=300)
                back_img    = gr.Image(label="Kameez Back — Optional",
                                       sources="upload", type="numpy", height=300)
                shalwar_img = gr.Image(label="Shalwar / Trouser — Optional",
                                       sources="upload", type="numpy", height=300)
                dupatta_img = gr.Image(label="Dupatta — Optional",
                                       sources="upload", type="numpy", height=300)
            gr.HTML("""<div class="tip-box" style="margin-top:12px;">
                <strong>Garment tips:</strong>
                Upload any pieces you have — AI will composite them automatically.
                Product photos (even with backgrounds) work fine —
                backgrounds are removed automatically.
                The more pieces you upload, the better the suit detection.
            </div>""")

    # ── Row 2: Controls ───────────────────────────────────────────────────────
    gr.HTML('<hr class="gold-div">')
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            gr.HTML("""
            <div style="padding:8px 0;">
                <div style="font-family:'Cormorant Garamond',serif;font-size:1rem;
                            color:var(--gold);letter-spacing:.06em;margin-bottom:8px;">
                    ✦ How it works
                </div>
                <div style="font-size:.82rem;color:var(--muted);line-height:1.9;">
                    1 &nbsp;·&nbsp; Upload your model photo + kameez front (required), plus any other pieces<br>
                    2 &nbsp;·&nbsp; Press <em>Generate</em><br>
                    3 &nbsp;·&nbsp; GPT-4o reads your design automatically — no typing needed<br>
                    4 &nbsp;·&nbsp; 5 high-quality 4K looks generated with varied pose & aspect ratios<br>
                    5 &nbsp;·&nbsp; Download any image with the ↓ button
                </div>
            </div>""")

        with gr.Column(scale=2):
            with gr.Row():
                seed_slider    = gr.Slider(label="Seed", minimum=0,
                                           maximum=MAX_SEED, step=1, value=0, scale=3)
                randomize_seed = gr.Checkbox(label="🎲 Random", value=True, scale=1)
            run_btn    = gr.Button("✦  Generate 5 Looks  ✦",
                                   elem_id="run-btn", variant="primary")
            seed_out   = gr.Number(label="Seed used", visible=False)
            status_out = gr.Textbox(label="Status", interactive=False,
                                    lines=3, elem_id="status-box")

    # ── Results ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <hr class="gold-div">
    <div class="sec-head" style="text-align:center;margin-bottom:4px;">✦ Generated Looks</div>
    <div style="text-align:center;font-size:.76rem;color:var(--muted);margin-bottom:20px;letter-spacing:.06em;">
        Standard · Free Flow · Portrait · Square · Reposed
    </div>""")

    with gr.Row(equal_height=True, elem_classes=["result-row"]):
        out1 = gr.Image(label="✦ Look 1 – Standard",    height=520, show_download_button=True)
        out2 = gr.Image(label="✦ Look 2 – Free Flow",   height=520, show_download_button=True)
        out3 = gr.Image(label="✦ Look 3 – Portrait",    height=520, show_download_button=True)
        out4 = gr.Image(label="✦ Look 4 – Square Crop", height=520, show_download_button=True)
        out5 = gr.Image(label="✦ Look 5 – Reposed",     height=520, show_download_button=True)

    gr.HTML('<div id="pk-footer">Powered by GPT-4o Vision · fal-ai/image-apps-v2/virtual-try-on · 4K Output</div>')

    run_btn.click(
        fn=generate,
        inputs=[person_img,
                front_img, back_img, shalwar_img, dupatta_img,
                seed_slider, randomize_seed],
        outputs=[out1, out2, out3, out4, out5, seed_out, status_out],
    )

Tryon.queue().launch()