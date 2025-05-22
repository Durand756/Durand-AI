import os
import io
import base64
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import DiffusionPipeline, AutoencoderKL
from PIL import Image
import uvicorn
from typing import Optional
import logging
import time
 
# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration du modèle
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI(title="AI Image Generator", description="Générateur d'images ultra-réalistes avec SDXL")

# Modèle de données pour les requêtes
class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry, bad quality, distorted, ugly, low resolution"
    steps: int = 30
    cfg_scale: float = 7.5
    seed: Optional[int] = None
    width: int = 1024
    height: int = 1024

# Variable globale pour le pipeline
pipeline = None

def load_model():
    """Charge le modèle Stable Diffusion XL"""
    global pipeline
    try:
        logger.info(f"Chargement du modèle sur {DEVICE}...")
        
        # Chargement du VAE optimisé
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=TORCH_DTYPE
        )
        
        # Chargement du pipeline principal
        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True
        )
        
        pipeline = pipeline.to(DEVICE)
        
        # Optimisations pour la performance
        if DEVICE == "cuda":
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_slicing()
            pipeline.enable_attention_slicing(1)
            
            # Compilation torch pour optimisation
            if hasattr(torch, 'compile'):
                pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
        
        logger.info("Modèle chargé avec succès!")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        return False

def generate_image(request: ImageRequest) -> str:
    """Génère une image à partir du prompt"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        # Configuration du générateur avec seed
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(request.seed)
        
        logger.info(f"Génération d'image: {request.prompt[:50]}...")
        start_time = time.time()
        
        # Génération de l'image
        with torch.inference_mode():
            result = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.steps,
                guidance_scale=request.cfg_scale,
                width=request.width,
                height=request.height,
                generator=generator
            )
        
        image = result.images[0]
        generation_time = time.time() - start_time
        logger.info(f"Image générée en {generation_time:.2f}s")
        
        # Conversion en base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de génération: {str(e)}")

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
    """Vérification du statut de l'API"""
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "device": DEVICE,
        "torch_version": torch.__version__
    }

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Page d'accueil avec l'interface web"""
    html_content = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Generator - Ultra-Réaliste</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .container-fluid {
            padding: 2rem;
        }
        .card {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            background: rgba(255,255,255,0.95);
        }
        .card-header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border-radius: 15px 15px 0 0 !important;
            text-align: center;
            padding: 1.5rem;
        }
        .btn-generate {
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            transition: all 0.3s ease;
        }
        .btn-generate:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .image-container {
            min-height: 300px;
            background: #f8f9fa;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        .generated-image {
            max-width: 100%;
            max-height: 500px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .loading {
            display: none;
        }
        .loading.active {
            display: block;
        }
        .form-range {
            accent-color: #667eea;
        }
        .badge {
            font-size: 0.8em;
        }
        .parameter-group {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row justify-content-center">
            <div class="col-12 col-xl-10">
                <div class="card">
                    <div class="card-header">
                        <h1 class="mb-0"><i class="fas fa-magic"></i> AI Image Generator Ultra-Réaliste</h1>
                        <p class="mb-0 mt-2">Powered by Stable Diffusion XL</p>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <!-- Panneau de contrôle -->
                            <div class="col-md-4">
                                <form id="imageForm">
                                    <div class="mb-3">
                                        <label for="prompt" class="form-label">
                                            <i class="fas fa-pencil-alt"></i> Prompt
                                        </label>
                                        <textarea class="form-control" id="prompt" rows="3" 
                                                placeholder="Décrivez l'image que vous voulez générer..." required></textarea>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label for="negativePrompt" class="form-label">
                                            <i class="fas fa-ban"></i> Prompt Négatif
                                        </label>
                                        <textarea class="form-control" id="negativePrompt" rows="2" 
                                                placeholder="Ce que vous ne voulez pas voir..."></textarea>
                                    </div>
                                    
                                    <div class="parameter-group">
                                        <h6><i class="fas fa-sliders-h"></i> Paramètres Avancés</h6>
                                        
                                        <div class="mb-3">
                                            <label for="steps" class="form-label">
                                                Étapes: <span id="stepsValue">30</span>
                                            </label>
                                            <input type="range" class="form-range" id="steps" min="10" max="100" value="30">
                                        </div>
                                        
                                        <div class="mb-3">
                                            <label for="cfgScale" class="form-label">
                                                CFG Scale: <span id="cfgValue">7.5</span>
                                            </label>
                                            <input type="range" class="form-range" id="cfgScale" min="1" max="20" step="0.5" value="7.5">
                                        </div>
                                        
                                        <div class="mb-3">
                                            <label for="seed" class="form-label">
                                                <i class="fas fa-dice"></i> Seed (optionnel)
                                            </label>
                                            <input type="number" class="form-control" id="seed" placeholder="Laissez vide pour aléatoire">
                                        </div>
                                        
                                        <div class="row">
                                            <div class="col-6">
                                                <label for="width" class="form-label">Largeur</label>
                                                <select class="form-select" id="width">
                                                    <option value="512">512px</option>
                                                    <option value="768">768px</option>
                                                    <option value="1024" selected>1024px</option>
                                                </select>
                                            </div>
                                            <div class="col-6">
                                                <label for="height" class="form-label">Hauteur</label>
                                                <select class="form-select" id="height">
                                                    <option value="512">512px</option>
                                                    <option value="768">768px</option>
                                                    <option value="1024" selected>1024px</option>
                                                </select>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <button type="submit" class="btn btn-generate btn-primary w-100">
                                        <i class="fas fa-magic"></i> Générer l'Image
                                    </button>
                                </form>
                            </div>
                            
                            <!-- Zone d'affichage de l'image -->
                            <div class="col-md-8">
                                <div class="image-container" id="imageContainer">
                                    <div class="text-center text-muted">
                                        <i class="fas fa-image fa-3x mb-3"></i>
                                        <p>Votre image générée apparaîtra ici</p>
                                    </div>
                                    
                                    <div class="loading" id="loading">
                                        <div class="text-center">
                                            <div class="spinner-border text-primary" role="status">
                                                <span class="visually-hidden">Génération en cours...</span>
                                            </div>
                                            <p class="mt-3">Génération en cours...</p>
                                            <small class="text-muted">Cela peut prendre quelques minutes</small>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="mt-3 text-center" id="downloadSection" style="display: none;">
                                    <button class="btn btn-success" id="downloadBtn">
                                        <i class="fas fa-download"></i> Télécharger PNG
                                    </button>
                                    <div class="mt-2">
                                        <small class="text-muted" id="imageInfo"></small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Variables globales
        let currentImageData = null;
        
        // Mise à jour des valeurs des sliders
        document.getElementById('steps').addEventListener('input', function() {
            document.getElementById('stepsValue').textContent = this.value;
        });
        
        document.getElementById('cfgScale').addEventListener('input', function() {
            document.getElementById('cfgValue').textContent = this.value;
        });
        
        // Gestion du formulaire
        document.getElementById('imageForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const prompt = document.getElementById('prompt').value;
            const negativePrompt = document.getElementById('negativePrompt').value;
            const steps = parseInt(document.getElementById('steps').value);
            const cfgScale = parseFloat(document.getElementById('cfgScale').value);
            const seed = document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null;
            const width = parseInt(document.getElementById('width').value);
            const height = parseInt(document.getElementById('height').value);
            
            if (!prompt.trim()) {
                alert('Veuillez entrer un prompt');
                return;
            }
            
            // Affichage du loading
            document.getElementById('loading').classList.add('active');
            document.getElementById('downloadSection').style.display = 'none';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        negative_prompt: negativePrompt,
                        steps: steps,
                        cfg_scale: cfgScale,
                        seed: seed,
                        width: width,
                        height: height
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayImage(data.image, data.parameters);
                } else {
                    throw new Error(data.error || 'Erreur inconnue');
                }
                
            } catch (error) {
                console.error('Erreur:', error);
                alert('Erreur lors de la génération: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        });
        
        // Affichage de l'image générée
        function displayImage(imageData, parameters) {
            currentImageData = imageData;
            
            const container = document.getElementById('imageContainer');
            container.innerHTML = `<img src="${imageData}" alt="Image générée" class="generated-image">`;
            
            // Affichage des informations
            const info = `${parameters.steps} étapes • CFG: ${parameters.cfg_scale} • ${parameters.size}`;
            if (parameters.seed) {
                info += ` • Seed: ${parameters.seed}`;
            }
            document.getElementById('imageInfo').textContent = info;
            document.getElementById('downloadSection').style.display = 'block';
        }
        
        // Téléchargement de l'image
        document.getElementById('downloadBtn').addEventListener('click', function() {
            if (currentImageData) {
                const link = document.createElement('a');
                link.download = `ai-generated-${Date.now()}.png`;
                link.href = currentImageData;
                link.click();
            }
        });
        
        // Prompts d'exemple
        const examplePrompts = [
            "A photorealistic portrait of a beautiful woman with flowing auburn hair, soft natural lighting, professional photography",
            "A majestic mountain landscape at sunrise, golden hour lighting, ultra-detailed, cinematic",
            "A futuristic cityscape at night, neon lights, cyberpunk style, highly detailed",
            "A cute fluffy cat sitting in a garden, soft natural lighting, adorable, high quality"
        ];
        
        // Ajout d'un bouton pour les exemples
        const promptTextarea = document.getElementById('prompt');
        promptTextarea.addEventListener('focus', function() {
            if (!this.value) {
                this.placeholder = examplePrompts[Math.floor(Math.random() * examplePrompts.length)];
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# Événement de démarrage
@app.on_event("startup")
async def startup_event():
    """Chargement du modèle au démarrage"""
    logger.info("Démarrage de l'application...")
    success = load_model()
    if not success:
        logger.warning("Le modèle n'a pas pu être chargé. Certaines fonctionnalités seront indisponibles.")

# Configuration pour Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
