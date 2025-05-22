#!/usr/bin/env python3
"""
Générateur d'Images AI Ultra-Réalistes - Version Production
Optimisé pour le déploiement sur Render.com
"""

import os
import io
import base64
import json
import uuid
import logging
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS

# Configuration du logging pour production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)

# Configuration de l'application
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

class ProductionImageGenerator:
    """Générateur d'images optimisé pour la production"""
    
    def __init__(self):
        self.device = "cpu"  # Force CPU pour Render
        self.pipeline = None
        self.model_loaded = False
        self.is_loading = False
        logger.info("Initialisation du générateur en mode production")
        
    def load_model_lazy(self):
        """Charge le modèle de façon paresseuse (au premier appel)"""
        if self.model_loaded or self.is_loading:
            return
            
        self.is_loading = True
        logger.info("Chargement du modèle Stable Diffusion...")
        
        try:
            # Import dynamique pour éviter les erreurs au startup
            import torch
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            
            # Modèle optimisé pour CPU
            model_id = "runwayml/stable-diffusion-v1-5"
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # CPU nécessite float32
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            # Optimisation pour CPU
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Optimisations mémoire pour CPU
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_sequential_cpu_offload()
            
            self.model_loaded = True
            logger.info("Modèle chargé avec succès en mode CPU")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            self.model_loaded = False
        finally:
            self.is_loading = False
    
    def generate_image(self, prompt, negative_prompt="", width=512, height=512, 
                      num_inference_steps=20, guidance_scale=7.5, seed=None):
        """Génère une image avec paramètres optimisés pour CPU"""
        
        # Chargement paresseux du modèle
        if not self.model_loaded:
            self.load_model_lazy()
            
        if not self.model_loaded:
            raise Exception("Modèle non disponible")
        
        try:
            import torch
            
            # Prompt optimisé
            enhanced_prompt = f"{prompt}, high quality, detailed"
            enhanced_negative = f"{negative_prompt}, low quality, blurry, distorted" if negative_prompt else "low quality, blurry"
            
            # Générateur pour reproductibilité
            generator = None
            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            
            # Génération avec paramètres optimisés CPU
            logger.info(f"Génération d'image: {prompt[:50]}...")
            
            with torch.no_grad():
                result = self.pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=enhanced_negative,
                    width=min(width, 512),  # Limiter pour CPU
                    height=min(height, 512),
                    num_inference_steps=min(num_inference_steps, 30),  # Limiter steps
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Erreur génération: {e}")
            raise

# Instance globale du générateur
image_generator = ProductionImageGenerator()

# Template HTML intégré (pour éviter les problèmes de fichiers)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Générateur d'Images AI - Production</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container {
            max-width: 1000px; margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px; padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        .header {
            text-align: center; margin-bottom: 30px;
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white; padding: 30px; margin: -30px -30px 30px -30px;
            border-radius: 20px 20px 0 0;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; font-weight: 300; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #2c3e50; }
        .form-group input, .form-group textarea, .form-group select {
            width: 100%; padding: 12px; border: 2px solid #e1e8ed;
            border-radius: 8px; font-size: 14px; transition: all 0.3s ease;
        }
        .form-group input:focus, .form-group textarea:focus {
            outline: none; border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .generate-btn {
            width: 100%; padding: 15px;
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white; border: none; border-radius: 10px;
            font-size: 16px; font-weight: 600; cursor: pointer;
            transition: all 0.3s ease;
        }
        .generate-btn:hover { transform: translateY(-2px); }
        .generate-btn:disabled { background: #95a5a6; cursor: not-allowed; transform: none; }
        .image-container {
            margin-top: 30px; text-align: center;
            border: 2px dashed #d1d8e0; border-radius: 15px;
            padding: 30px; min-height: 300px;
            display: flex; align-items: center; justify-content: center;
        }
        .generated-image { max-width: 100%; border-radius: 10px; }
        .loading { display: none; }
        .loading.show { display: block; }
        .spinner {
            width: 40px; height: 40px; border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db; border-radius: 50%;
            animation: spin 1s linear infinite; margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .status { padding: 10px; border-radius: 6px; margin-bottom: 20px; text-align: center; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        @media (max-width: 768px) {
            .form-row { grid-template-columns: 1fr; }
            .header h1 { font-size: 2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Générateur d'Images AI</h1>
            <p>Version Production - Render.com</p>
        </div>

        <div id="status" class="status" style="display: none;"></div>

        <form id="generateForm">
            <div class="form-group">
                <label for="prompt">Description de l'image ✨</label>
                <textarea id="prompt" rows="3" placeholder="Décrivez l'image que vous souhaitez générer..." required></textarea>
            </div>

            <div class="form-group">
                <label for="negativePrompt">Éléments à éviter (optionnel)</label>
                <textarea id="negativePrompt" rows="2" placeholder="Ex: flou, basse qualité, déformé..."></textarea>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label for="width">Largeur</label>
                    <select id="width">
                        <option value="256">256px (Rapide)</option>
                        <option value="512" selected>512px (Standard)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="height">Hauteur</label>
                    <select id="height">
                        <option value="256">256px (Rapide)</option>
                        <option value="512" selected>512px (Standard)</option>
                    </select>
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label for="steps">Étapes de génération</label>
                    <select id="steps">
                        <option value="10">10 (Ultra rapide)</option>
                        <option value="15">15 (Rapide)</option>
                        <option value="20" selected>20 (Standard)</option>
                        <option value="25">25 (Qualité)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="seed">Seed (optionnel)</label>
                    <input type="number" id="seed" placeholder="Aléatoire">
                </div>
            </div>

            <button type="submit" class="generate-btn" id="generateBtn">
                <span class="btn-text">Générer l'Image 🚀</span>
            </button>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px;">Génération en cours... Cela peut prendre 1-2 minutes</p>
        </div>

        <div class="image-container" id="imageContainer">
            <div>
                <div style="font-size: 48px; margin-bottom: 10px;">🎨</div>
                <div>Votre image générée apparaîtra ici</div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('generateForm');
        const statusDiv = document.getElementById('status');
        const loadingDiv = document.getElementById('loading');
        const imageContainer = document.getElementById('imageContainer');
        const generateBtn = document.getElementById('generateBtn');

        function showStatus(message, type) {
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            statusDiv.style.display = 'block';
            setTimeout(() => statusDiv.style.display = 'none', 5000);
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                showStatus('Veuillez entrer une description', 'error');
                return;
            }

            generateBtn.disabled = true;
            loadingDiv.classList.add('show');
            showStatus('Génération en cours...', 'info');

            const params = {
                prompt: prompt,
                negative_prompt: document.getElementById('negativePrompt').value,
                width: parseInt(document.getElementById('width').value),
                height: parseInt(document.getElementById('height').value),
                num_inference_steps: parseInt(document.getElementById('steps').value),
                seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null
            };

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });

                const result = await response.json();

                if (result.success) {
                    imageContainer.innerHTML = `<img src="${result.image}" alt="Generated" class="generated-image">`;
                    showStatus('Image générée avec succès!', 'success');
                } else {
                    throw new Error(result.error || 'Erreur de génération');
                }

            } catch (error) {
                console.error('Erreur:', error);
                showStatus(`Erreur: ${error.message}`, 'error');
                imageContainer.innerHTML = `
                    <div>
                        <div style="font-size: 48px; margin-bottom: 10px;">❌</div>
                        <div>Erreur de génération</div>
                    </div>
                `;
            } finally {
                generateBtn.disabled = false;
                loadingDiv.classList.remove('show');
            }
        });

        // Vérifier le statut au chargement
        fetch('/api/status')
            .then(r => r.json())
            .then(data => {
                if (data.model_loaded) {
                    showStatus('Modèle prêt!', 'success');
                } else {
                    showStatus('Modèle en cours de chargement...', 'info');
                }
            })
            .catch(() => showStatus('Erreur de connexion', 'error'));
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Page d'accueil avec template intégré"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """API de génération d'images"""
    try:
        data = request.get_json()
        
        # Validation
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt requis'}), 400
        
        if len(prompt) > 500:
            return jsonify({'error': 'Prompt trop long (max 500 caractères)'}), 400
        
        # Paramètres avec limites pour CPU
        params = {
            'prompt': prompt,
            'negative_prompt': data.get('negative_prompt', ''),
            'width': min(int(data.get('width', 512)), 512),
            'height': min(int(data.get('height', 512)), 512),
            'num_inference_steps': min(int(data.get('num_inference_steps', 20)), 30),
            'guidance_scale': min(float(data.get('guidance_scale', 7.5)), 15.0),
            'seed': data.get('seed')
        }
        
        logger.info(f"Génération demandée: {prompt[:50]}...")
        
        # Génération
        image = image_generator.generate_image(**params)
        
        # Conversion en base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info("Image générée avec succès")
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{image_base64}",
            'parameters': params
        })
        
    except Exception as e:
        logger.error(f"Erreur génération: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Status de l'API"""
    return jsonify({
        'status': 'online',
        'model_loaded': image_generator.model_loaded,
        'is_loading': image_generator.is_loading,
        'device': image_generator.device,
        'environment': 'production',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    """Health check pour Render"""
    return jsonify({'status': 'healthy'}), 200

# Point d'entrée pour Gunicorn
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"Démarrage de l'application sur le port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

# Application WSGI pour Gunicorn
application = app
