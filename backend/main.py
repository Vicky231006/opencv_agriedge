import os
import io
import cv2
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import SegformerForSemanticSegmentation

# Initialize FastAPI framework
app = FastAPI(title="HERMES Agri-Edge Core Inference API", version="1.0")

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine targeted hardware accelerator context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Synchronizing backend runtime execution on target device: {device}")

# =====================================================================
# 1. PIECE TOGETHER COMPONENT ARCHITECTURES
# =====================================================================
class SegFormerDecoderHead(nn.Module):
    def __init__(self, embedding_dims=[32, 64, 160, 256], num_classes=2):
        super(SegFormerDecoderHead, self).__init__()
        self.linear_c4 = nn.Linear(embedding_dims[3], 128)
        self.linear_c3 = nn.Linear(embedding_dims[2], 128)
        self.linear_c2 = nn.Linear(embedding_dims[1], 128)
        self.linear_c1 = nn.Linear(embedding_dims[0], 128)
        
        self.linear_fuse = nn.Conv2d(128 * 4, 128, kernel_size=1)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, c1, c2, c3, c4, target_shape=(224, 224)):
        B, _, H, W = c1.shape
        
        _c4 = self.linear_c4(c4.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=(H, W), mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=(H, W), mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=(H, W), mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, -1, c1.shape[2], c1.shape[3])
        
        x = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        x = F.relu(self.linear_fuse(x))
        logits = self.classifier(x)
        
        return F.interpolate(logits, size=target_shape, mode='bilinear', align_corners=False)


class YieldRiskCore:
    @staticmethod
    def calculate_yield_decay(prediction_tensor):
        total_lesion_pixels = torch.sum(prediction_tensor == 1).item()
        total_background_pixels = torch.sum(prediction_tensor == 0).item()
        total_surface = total_background_pixels + total_lesion_pixels
        
        if total_surface == 0:
            return 0.0
        damage_ratio = total_lesion_pixels / total_surface
        estimated_yield_loss = 1.0 - np.exp(-3.5 * damage_ratio)
        return float(estimated_yield_loss * 100)


class AgriManagementEngine:
    REGISTRY = {
        "tomato": {
            "name": "Tomato (Solanum lycopersicum)",
            "chemical": "Propiconazole 25% EC",
            "mild_directive": "Isolate localized spot. Trigger variable-rate micro-sprayer intervention to limit early blight vectors.",
            "critical_directive": "CRITICAL: Severe systemic infection. Trigger immediate crop quarantine. Deploy heavy chemical shielding."
        },
        "apple": {
            "name": "Apple (Malus domestica)",
            "chemical": "Captan 80 WDG Fungicide",
            "mild_directive": "Apply localized orchard misting sequence. Monitor air moisture values over next 48 hours to prevent powdery mildew spread.",
            "critical_directive": "CRITICAL: Scab/Rot threat cross-contamination risk high. Execute macro-orchard chemical shield immediately."
        },
        "grape": {
            "name": "Grape (Vitis vinifera)",
            "chemical": "Copper Sulfate / Bordeaux Mixture",
            "mild_directive": "Deploy vineyard canopy micro-sprayer. Fine-tune ground irrigation rates to halt downy mildew incubation.",
            "critical_directive": "CRITICAL: High risk of total canopy cluster rot. Isolate vineyard row block and apply maximum legal dosage."
        },
        "potato": {
            "name": "Potato (Solanum tuberosum)",
            "chemical": "Mancozeb 75% WP",
            "mild_directive": "Execute early patch dusting run. Lower soil moisture targets to prevent subsurface spore migration.",
            "critical_directive": "CRITICAL: Late Blight vector identified. Initiate immediate wide-area field desiccation sequence to protect remaining yield fields."
        }
    }

    @classmethod
    def generate_action_prescription(cls, yield_loss_percentage, crop_type):
        crop = crop_type.lower() if crop_type.lower() in cls.REGISTRY else "tomato"
        spec = cls.REGISTRY[crop]
        
        if yield_loss_percentage == 0.0:
            return {
                "alarm_level": f"🟢 HEALTHY {spec['name'].upper()}",
                "intervention_directive": f"No active disease lesions mapped across this {spec['name']} profile. Maintain baseline automated drip irrigation.",
                "pesticide_dosage_ml_per_m2": 0.0,
                "chemical_composition": "None (Organic Balance)"
            }
        elif 0.0 < yield_loss_percentage <= 20.0:
            return {
                "alarm_level": f"🟡 MILD {crop.upper()} THREAT CONTEXT",
                "intervention_directive": spec["mild_directive"],
                "pesticide_dosage_ml_per_m2": 7.5,
                "chemical_composition": spec["chemical"]
            }
        else:
            return {
                "alarm_level": f"🔴 CRITICAL {crop.upper()} VECTOR EMERGENCY",
                "intervention_directive": spec["critical_directive"],
                "pesticide_dosage_ml_per_m2": 32.0,
                "chemical_composition": spec["chemical"]
            }

# =====================================================================
# 2. RUNTIME WEIGHT LOADING SEQUENCER
# =====================================================================
WEIGHTS_PATH = "HERMES_AgriEdge_v1.pth"

# Instantiate structures globally right away
backbone = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True
).to(device)

model = SegFormerDecoderHead(embedding_dims=[32, 64, 160, 256], num_classes=2).to(device)

@app.on_event("startup")
def load_production_pipeline():
    global backbone, model
    print("⏳ Synchronizing model configurations and loading serialized weights matrix...")
    
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"❌ Model check-point file missing! Please place your downloaded '{WEIGHTS_PATH}' file in the server root folder."
        )
    
    # Map checkpoint states directly onto layers
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    model.load_state_dict(checkpoint['decoder_head_state_dict'])
    
    backbone.eval()
    model.eval()
    print("🔥 Systems synchronized and active. PyTorch inference layers online!")

# =====================================================================
# 3. INTERACTIVE CORRELATION INFERENCE INTERFACE
# =====================================================================
@app.post("/api/analyze")
async def process_canopy_matrix(
    file: UploadFile = File(...),
    crop_type: str = Form("tomato")
):
    global backbone, model
    
    # Extract file bytes buffer and convert to raw imaging matrix
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_raw is None:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid media file buffer format decoded.")
        
    orig_h, orig_w = img_raw.shape[:2]
    
    # Resizing transformations to match deep SegFormer attention matrices
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Run forward inference loop across PyTorch layers
    with torch.no_grad():
        backbone_features = backbone(img_tensor, output_hidden_states=True).hidden_states
        logits = model(*backbone_features, target_shape=(224, 224))
        prediction_mask = torch.argmax(logits, dim=1).squeeze(0)
        
    # Calculate continuous spatial risk parameters
    risk_percentage = YieldRiskCore.calculate_yield_decay(prediction_mask)
    prescription_data = AgriManagementEngine.generate_action_prescription(risk_percentage, crop_type)
    
    # Compute operational statistics metrics payload
    total_lesion_px = torch.sum(prediction_mask == 1).item()
    total_leaf_px = torch.sum(prediction_mask == 0).item() + total_lesion_px
    infection_ratio = total_lesion_px / total_leaf_px if total_leaf_px > 0 else 0.0
    
    # Create the color-mapped visual PNG mask asset
    mask_np = prediction_mask.cpu().numpy().astype(np.uint8)
    visual_mask = np.zeros((224, 224, 3), dtype=np.uint8)
    visual_mask[mask_np == 1] = [180, 255, 0]  # Electric Lime
    
    # Scaling predictions back to original coordinate frames
    visual_mask_resized = cv2.resize(visual_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(visual_mask_resized, cv2.COLOR_RGB2BGR))
    base64_encoded_mask = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "status": "success",
        "analytics": {
            "detected_lesion_pixels": total_lesion_px,
            "total_leaf_pixels": total_leaf_px,
            "infection_ratio": float(infection_ratio),
            "calculated_yield_risk_percentage": float(risk_percentage)
        },
        "management_prescription": prescription_data,
        "mask_image": base64_encoded_mask
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)