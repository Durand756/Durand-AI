import os
import io
import base64
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import uvicorn
from typing import Optional
import logging
import time
import gc

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration optimisée pour CPU et mémoire limitée
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Modèle plus léger
DEVICE = "cpu"  # Force CPU pour Render gratuit
TORCH_DTYPE = torch.float32

app = FastAPI(title="AI Image Generator", description="Générateur d'images avec Stable Diffusion")

# Modèle de données pour les requêtes
class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry, bad quality, distorted"
    steps: int = 20  # Réduit pour performance CPU
    cfg_scale: float = 7.5
    seed: Optional[int] = None
    width: int = 512  # Réduit pour économiser RAM
    height: int = 512

# Variable globale pour le pipeline
pipeline = None

def load_model():
    """Charge le modèle Stable Diffusion optimisé CPU"""
    global pipeline
    try:
        logger.info("Chargement du modèle optimisé CPU...")
        
        # Chargement avec optimisations mémoire
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True,
            safety_checker=None,  # Désactiver pour économiser RAM
            requires_safety_checker=False
        )
        
        pipeline = pipeline.to(DEVICE)
        
        # Optimisations CPU
        pipeline.enable_attention_slicing(1)
        pipeline.enable_vae_slicing()
        
        # Optimisation mémoire agressive
        if hasattr(pipeline, 'enable_model_cpu_offload'):
            pipeline.enable_model_cpu_offload()
            
        logger.info("Modèle chargé avec succès!")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        return False

def generate_image(request: ImageRequest) -> str:
    """Génère une image avec optimisations mémoire"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        # Nettoyage mémoire avant génération
        gc.collect()
        
        # Configuration du générateur
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(request.seed)
        
        logger.info(f"Génération CPU: {request.prompt[:50]}...")
        start_time = time.time()
        
        # Génération avec paramètres optimisés
        with torch.inference_mode():
            result = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=min(request.steps, 25),  # Limite pour CPU
                guidance_scale=request.cfg_scale,
                width=min(request.width, 512),  # Limite pour RAM
                height=min(request.height, 512),
                generator=generator
            )
        
        image = result.images[0]
        generation_time = time.time() - start_time
        logger.info(f"Image générée en {generation_time:.2f}s")
        
        # Conversion optimisée
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Nettoyage mémoire après génération
        del result
        gc.collect()
        
        return img_base64
        
    except Exception as e:
        logger.error(f"Erreur génération: {e}")
        gc.collect()  # Nettoyage en cas d'erreur
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

# Routes API
@app.post("/api/generate")
async def api_generate_image(request: ImageRequest):
    """API endpoint pour générer une image"""
    try:
        image_base64 = generate_image(request)
        return {
            "success": True,
            "image": f"data:image/png;base64,{image_base64}",
            "prompt": request.prompt,
            "parameters": {
                "steps": request.steps,
                "cfg_scale": request.cfg_scale,
                "seed": request.seed,
                "size": f"{request.width}x{request.height}"
            }
        }
    except HTTPException as e:
        return {"success": False, "error": e.detail}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/status")
async def api_status():
    """Statut de l'API"""
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "device": DEVICE,
        "model": MODEL_ID,
        "mode": "CPU optimized"
    }

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Interface web optimisée"""
    html_content = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Generator - CPU Optimized</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .card {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            background: rgba(255,255,255,0.95);
        }
        .card-header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border-radius: 15px 15px 0 0 !important;
        }
        .btn-generate {
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            border-radius: 25px;
        }
        .image-container {
            min-height: 300px;
            background: #f8f9fa;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .generated-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
        }
        .loading { display: none; }
        .loading.active { display: block; }
        .cpu-warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container-fluid p-4">
        <div class="row justify-content-center">
            <div class="col-12 col-xl-10">
                <div class="card">
                    <div class="card-header text-center">
                        <h1><i class="fas fa-magic"></i> AI Image Generator</h1>
                        <p class="mb-0">CPU Optimized - Stable Diffusion v1.5</p>
                    </div>
                    <div class="card-body">
                        <div class="cpu-warning">
                            <i class="fas fa-info-circle"></i>
                            <strong>Mode CPU:</strong> Génération plus lente (2-5 min) mais gratuite. 
                            Résolution limitée à 512x512 pour optimiser les performances.
                        </div>
                        
                        <div class="row">
                            <div class="col-md-4">
                                <form id="imageForm">
                                    <div class="mb-3">
                                        <label class="form-label">
                                            <i class="fas fa-pencil-alt"></i> Prompt
                                        </label>
                                        <textarea class="form-control" id="prompt" rows="3" 
                                                placeholder="Décrivez votre image..." required></textarea>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">
                                            <i class="fas fa-ban"></i> Prompt Négatif
                                        </label>
                                        <textarea class="form-control" id="negativePrompt" rows="2"
                                                value="blurry, bad quality, distorted"></textarea>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">
                                            Étapes: <span id="stepsValue">20</span>
                                        </label>
                                        <input type="range" class="form-range" id="steps" 
                                               min="10" max="25" value="20">
                                        <small class="text-muted">Limité à 25 pour performance CPU</small>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">
                                            CFG Scale: <span id="cfgValue">7.5</span>
                                        </label>
                                        <input type="range" class="form-range" id="cfgScale" 
                                               min="1" max="15" step="0.5" value="7.5">
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">
                                            <i class="fas fa-dice"></i> Seed (optionnel)
                                        </label>
                                        <input type="number" class="form-control" id="seed" 
                                               placeholder="Aléatoire">
                                    </div>
                                    
                                    <button type="submit" class="btn btn-generate btn-primary w-100">
                                        <i class="fas fa-magic"></i> Générer Image
                                    </button>
                                </form>
                            </div>
                            
                            <div class="col-md-8">
                                <div class="image-container" id="imageContainer">
                                    <div class="text-center text-muted">
                                        <i class="fas fa-image fa-3x mb-3"></i>
                                        <p>Votre image apparaîtra ici</p>
                                        <small>Résolution: 512x512 pixels</small>
                                    </div>
                                    
                                    <div class="loading text-center" id="loading">
                                        <div class="spinner-border text-primary mb-3"></div>
                                        <p>Génération en cours...</p>
                                        <small class="text-muted">
                                            Mode CPU: Patience, cela peut prendre 2-5 minutes
                                        </small>
                                    </div>
                                </div>
                                
                                <div class="mt-3 text-center" id="downloadSection" style="display: none;">
                                    <button class="btn btn-success" id="downloadBtn">
                                        <i class="fas fa-download"></i> Télécharger PNG
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentImageData = null;
        
        // Mise à jour des valeurs
        document.getElementById('steps').addEventListener('input', function() {
            document.getElementById('stepsValue').textContent = this.value;
        });
        
        document.getElementById('cfgScale').addEventListener('input', function() {
            document.getElementById('cfgValue').textContent = this.value;
        });
        
        // Gestion du formulaire
        document.getElementById('imageForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                prompt: document.getElementById('prompt').value,
                negative_prompt: document.getElementById('negativePrompt').value,
                steps: parseInt(document.getElementById('steps').value),
                cfg_scale: parseFloat(document.getElementById('cfgScale').value),
                seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
                width: 512,
                height: 512
            };
            
            if (!formData.prompt.trim()) {
                alert('Veuillez entrer un prompt');
                return;
            }
            
            // Affichage loading
            document.getElementById('loading').classList.add('active');
            document.getElementById('downloadSection').style.display = 'none';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayImage(data.image);
                } else {
                    throw new Error(data.error);
                }
                
            } catch (error) {
                console.error('Erreur:', error);
                alert('Erreur: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        });
        
        function displayImage(imageData) {
            currentImageData = imageData;
            const container = document.getElementById('imageContainer');
            container.innerHTML = `<img src="${imageData}" alt="Generated" class="generated-image">`;
            document.getElementById('downloadSection').style.display = 'block';
        }
        
        document.getElementById('downloadBtn').addEventListener('click', function() {
            if (currentImageData) {
                const link = document.createElement('a');
                link.download = `ai-generated-${Date.now()}.png`;
                link.href = currentImageData;
                link.click();
            }
        });
        
        // Prompts d'exemple
        const examples = [
            "A beautiful sunset over mountains, golden hour, photorealistic",
            "A cute cat sitting in a garden, soft lighting, detailed",
            "A futuristic city, neon lights, cyberpunk style",
            "Portrait of a woman, professional photography, studio lighting"
        ];
        
        document.getElementById('prompt').addEventListener('focus', function() {
            if (!this.value) {
                this.placeholder = examples[Math.floor(Math.random() * examples.length)];
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.on_event("startup")
async def startup_event():
    """Chargement du modèle au démarrage"""
    logger.info("Démarrage optimisé CPU...")
    success = load_model()
    if not success:
        logger.warning("Modèle non chargé - fonctionnalités limitées")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main-cpu:app", host="0.0.0.0", port=port, reload=False)
