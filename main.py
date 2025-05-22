from flask import Flask, render_template, request, jsonify, send_file
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io
import base64
import os
import gc
import logging
from datetime import datetime
import threading
import time

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class AIImageGenerator:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Charge le modèle Stable Diffusion optimisé"""
        try:
            logger.info("Chargement du modèle Stable Diffusion...")
            
            # Utilisation d'un modèle plus léger pour Render gratuit
            model_id = "runwayml/stable-diffusion-v1-5"
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Optimisations pour performances
            self.pipe = self.pipe.to(self.device)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Optimisations mémoire
            if self.device == "cuda":
                self.pipe.enable_memory_efficient_attention()
                self.pipe.enable_xformers_memory_efficient_attention()
            
            # Pour CPU, utiliser moins de mémoire
            if self.device == "cpu":
                self.pipe.enable_attention_slicing()
            
            self.model_loaded = True
            logger.info(f"Modèle chargé avec succès sur {self.device}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            self.model_loaded = False
    
    def generate_image(self, prompt, negative_prompt="", width=512, height=512, 
                      steps=20, guidance_scale=7.5, seed=None):
        """Génère une image à partir d'un prompt"""
        if not self.model_loaded:
            raise Exception("Modèle non chargé")
        
        try:
            # Nettoyage mémoire
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Configuration du générateur
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Génération de l'image
            logger.info(f"Génération: '{prompt[:50]}...'")
            
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            image = result.images[0]
            
            # Conversion en base64 pour envoi web
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', quality=95)
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Nettoyage mémoire
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Erreur génération: {e}")
            raise e

# Instance globale du générateur
generator = AIImageGenerator()

@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """API de génération d'images"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        width = int(data.get('width', 512))
        height = int(data.get('height', 512))
        steps = int(data.get('steps', 20))
        guidance_scale = float(data.get('guidance_scale', 7.5))
        seed = data.get('seed')
        
        if seed:
            seed = int(seed)
        
        if not prompt:
            return jsonify({'error': 'Prompt requis'}), 400
        
        # Validation des paramètres
        width = min(max(width, 256), 1024)
        height = min(max(height, 256), 1024)
        steps = min(max(steps, 5), 50)
        guidance_scale = min(max(guidance_scale, 1.0), 20.0)
        
        # Génération
        start_time = time.time()
        image_base64 = generator.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        generation_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'generation_time': round(generation_time, 2),
            'parameters': {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'steps': steps,
                'guidance_scale': guidance_scale,
                'seed': seed
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Statut du générateur"""
    return jsonify({
        'model_loaded': generator.model_loaded,
        'device': generator.device,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    })

# Template HTML intégré
@app.before_first_request
def create_templates():
    """Crée le dossier templates et le fichier HTML"""
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Générateur d'Images AI Ultra-Réalistes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .image-preview {
            border: 2px dashed #dee2e6;
            border-radius: 15px;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        .image-preview.loading {
            border-color: #007bff;
            background: linear-gradient(45deg, #f8f9fa, #e9ecef);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .generated-image {
            max-width: 100%;
            max-height: 500px;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .btn-primary {
            background: linear-gradient(45deg, #007bff, #0056b3);
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            transition: all 0.3s ease;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,123,255,0.4);
        }
        .form-control, .form-select {
            border-radius: 10px;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
        }
        .form-control:focus, .form-select:focus {
            border-color: #007bff;
            box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
        }
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        .progress {
            height: 8px;
            border-radius: 10px;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="main-container p-4">
            <div class="text-center mb-5">
                <h1 class="display-4 fw-bold text-primary mb-3">
                    <i class="fas fa-magic"></i> Générateur d'Images AI
                </h1>
                <p class="lead text-muted">Créez des images ultra-réalistes avec l'intelligence artificielle</p>
            </div>

            <div class="row">
                <!-- Panneau de contrôle -->
                <div class="col-lg-4 mb-4">
                    <div class="card h-100">
                        <div class="card-header bg-primary text-white">
                            <h5 class="mb-0"><i class="fas fa-sliders-h"></i> Paramètres</h5>
                        </div>
                        <div class="card-body">
                            <form id="generateForm">
                                <div class="mb-3">
                                    <label class="form-label fw-bold">Prompt (Description)</label>
                                    <textarea id="prompt" class="form-control" rows="4" 
                                        placeholder="Ex: Portrait photoréaliste d'une femme avec des yeux bleus, éclairage naturel, haute définition"></textarea>
                                </div>

                                <div class="mb-3">
                                    <label class="form-label fw-bold">Prompt Négatif</label>
                                    <textarea id="negativePrompt" class="form-control" rows="2" 
                                        placeholder="Ex: flou, déformé, mauvaise qualité"></textarea>
                                </div>

                                <div class="row mb-3">
                                    <div class="col-6">
                                        <label class="form-label">Largeur</label>
                                        <select id="width" class="form-select">
                                            <option value="512" selected>512px</option>
                                            <option value="768">768px</option>
                                            <option value="1024">1024px</option>
                                        </select>
                                    </div>
                                    <div class="col-6">
                                        <label class="form-label">Hauteur</label>
                                        <select id="height" class="form-select">
                                            <option value="512" selected>512px</option>
                                            <option value="768">768px</option>
                                            <option value="1024">1024px</option>
                                        </select>
                                    </div>
                                </div>

                                <div class="mb-3">
                                    <label class="form-label">Étapes de génération</label>
                                    <input type="range" id="steps" class="form-range" min="5" max="50" value="20">
                                    <div class="d-flex justify-content-between">
                                        <small>5</small>
                                        <span id="stepsValue" class="fw-bold">20</span>
                                        <small>50</small>
                                    </div>
                                </div>

                                <div class="mb-3">
                                    <label class="form-label">Guidance Scale</label>
                                    <input type="range" id="guidanceScale" class="form-range" min="1" max="20" value="7.5" step="0.5">
                                    <div class="d-flex justify-content-between">
                                        <small>1.0</small>
                                        <span id="guidanceValue" class="fw-bold">7.5</span>
                                        <small>20.0</small>
                                    </div>
                                </div>

                                <div class="mb-4">
                                    <label class="form-label">Seed (optionnel)</label>
                                    <input type="number" id="seed" class="form-control" 
                                        placeholder="Laissez vide pour aléatoire">
                                </div>

                                <button type="submit" id="generateBtn" class="btn btn-primary w-100 fw-bold">
                                    <i class="fas fa-magic"></i> Générer l'Image
                                </button>
                            </form>
                        </div>
                    </div>
                </div>

                <!-- Zone d'affichage -->
                <div class="col-lg-8">
                    <div class="card h-100">
                        <div class="card-header bg-success text-white">
                            <h5 class="mb-0"><i class="fas fa-image"></i> Résultat</h5>
                        </div>
                        <div class="card-body">
                            <div id="imagePreview" class="image-preview mb-3">
                                <div class="text-center text-muted">
                                    <i class="fas fa-cloud-upload-alt fa-3x mb-3"></i>
                                    <h5>Votre image apparaîtra ici</h5>
                                    <p>Entrez un prompt et cliquez sur "Générer"</p>
                                </div>
                            </div>

                            <div id="progressContainer" class="d-none mb-3">
                                <div class="progress">
                                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                         role="progressbar" style="width: 100%"></div>
                                </div>
                                <div class="text-center mt-2">
                                    <small class="text-muted">Génération en cours...</small>
                                </div>
                            </div>

                            <div id="imageInfo" class="d-none">
                                <div class="row text-center">
                                    <div class="col-md-6">
                                        <strong>Temps de génération:</strong>
                                        <span id="generationTime" class="text-primary"></span>
                                    </div>
                                    <div class="col-md-6">
                                        <button id="downloadBtn" class="btn btn-outline-primary btn-sm">
                                            <i class="fas fa-download"></i> Télécharger
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Exemples de prompts -->
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header bg-info text-white">
                            <h5 class="mb-0"><i class="fas fa-lightbulb"></i> Exemples de Prompts</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6 class="fw-bold">Portraits:</h6>
                                    <ul class="list-unstyled">
                                        <li class="prompt-example mb-2 p-2 bg-light rounded cursor-pointer">
                                            "Portrait photoréaliste d'une femme aux cheveux bruns, sourire naturel, éclairage doux"
                                        </li>
                                        <li class="prompt-example mb-2 p-2 bg-light rounded cursor-pointer">
                                            "Homme âgé avec barbe blanche, regard sage, photographie noir et blanc"
                                        </li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6 class="fw-bold">Paysages:</h6>
                                    <ul class="list-unstyled">
                                        <li class="prompt-example mb-2 p-2 bg-light rounded cursor-pointer">
                                            "Coucher de soleil sur montagne, ciel coloré, lac en premier plan, ultra détaillé"
                                        </li>
                                        <li class="prompt-example mb-2 p-2 bg-light rounded cursor-pointer">
                                            "Forêt tropicale luxuriante, rayons de soleil, brume matinale, hyperréaliste"
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Mise à jour des valeurs des sliders
        document.getElementById('steps').addEventListener('input', function() {
            document.getElementById('stepsValue').textContent = this.value;
        });

        document.getElementById('guidanceScale').addEventListener('input', function() {
            document.getElementById('guidanceValue').textContent = this.value;
        });

        // Gestion des exemples de prompts
        document.querySelectorAll('.prompt-example').forEach(function(element) {
            element.style.cursor = 'pointer';
            element.addEventListener('click', function() {
                document.getElementById('prompt').value = this.textContent.replace(/"/g, '');
            });
        });

        // Gestion du formulaire
        document.getElementById('generateForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const btn = document.getElementById('generateBtn');
            const preview = document.getElementById('imagePreview');
            const progress = document.getElementById('progressContainer');
            const info = document.getElementById('imageInfo');
            
            // Validation
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Veuillez entrer un prompt de description');
                return;
            }
            
            // Interface de chargement
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Génération...';
            preview.className = 'image-preview loading';
            preview.innerHTML = '<div class="text-center"><div class="spinner-border text-primary"></div><p class="mt-3">Génération en cours...</p></div>';
            progress.classList.remove('d-none');
            info.classList.add('d-none');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        negative_prompt: document.getElementById('negativePrompt').value,
                        width: parseInt(document.getElementById('width').value),
                        height: parseInt(document.getElementById('height').value),
                        steps: parseInt(document.getElementById('steps').value),
                        guidance_scale: parseFloat(document.getElementById('guidanceScale').value),
                        seed: document.getElementById('seed').value || null
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Affichage de l'image
                    preview.className = 'image-preview';
                    preview.innerHTML = `<img src="data:image/png;base64,${result.image}" class="generated-image" alt="Image générée">`;
                    
                    // Informations
                    document.getElementById('generationTime').textContent = `${result.generation_time}s`;
                    
                    // Bouton de téléchargement
                    document.getElementById('downloadBtn').onclick = function() {
                        const link = document.createElement('a');
                        link.href = `data:image/png;base64,${result.image}`;
                        link.download = `ai_generated_${Date.now()}.png`;
                        link.click();
                    };
                    
                    info.classList.remove('d-none');
                } else {
                    throw new Error(result.error || 'Erreur inconnue');
                }
                
            } catch (error) {
                preview.className = 'image-preview';
                preview.innerHTML = `<div class="text-center text-danger"><i class="fas fa-exclamation-triangle fa-3x mb-3"></i><h5>Erreur</h5><p>${error.message}</p></div>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-magic"></i> Générer l\'Image';
                progress.classList.add('d-none');
            }
        });

        // Vérification du statut au chargement
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                console.log('Statut du générateur:', data);
                if (!data.model_loaded) {
                    document.getElementById('imagePreview').innerHTML = 
                        '<div class="text-center text-warning"><i class="fas fa-exclamation-triangle fa-3x mb-3"></i><h5>Chargement du modèle...</h5><p>Veuillez patienter quelques instants</p></div>';
                }
            })
            .catch(error => console.error('Erreur de statut:', error));
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
