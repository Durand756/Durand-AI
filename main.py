from flask import Flask, render_template, request, jsonify, send_file
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io
import base64
import os
import gc
import threading
import time
from datetime import datetime

app = Flask(__name__)

# Configuration globale
class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = None
        self.loading = False
        self.loaded = False

config = Config()

def load_model():
    """Charge le modèle Stable Diffusion de manière asynchrone"""
    config.loading = True
    try:
        print(f"Chargement du modèle sur {config.device}...")
        
        # Chargement du pipeline
        config.pipe = StableDiffusionPipeline.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Optimisation du scheduler
        config.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            config.pipe.scheduler.config
        )
        
        # Déplacement vers le device approprié
        config.pipe = config.pipe.to(config.device)
        
        # Optimisations pour CPU/GPU
        if config.device == "cuda":
            config.pipe.enable_memory_efficient_attention()
            config.pipe.enable_attention_slicing()
        else:
            # Optimisations pour CPU
            config.pipe.enable_attention_slicing(1)
        
        config.loaded = True
        config.loading = False
        print("Modèle chargé avec succès!")
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        config.loading = False
        config.loaded = False

def generate_image(prompt, negative_prompt="", steps=20, guidance_scale=7.5, width=512, height=512):
    """Génère une image à partir d'un prompt"""
    if not config.loaded:
        return None, "Modèle non chargé"
    
    try:
        # Nettoyage de la mémoire
        if config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Génération de l'image
        with torch.no_grad():
            result = config.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=torch.Generator(device=config.device).manual_seed(int(time.time()))
            )
        
        image = result.images[0]
        
        # Conversion en base64 pour l'affichage web
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64, None
        
    except Exception as e:
        return None, f"Erreur lors de la génération: {str(e)}"

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """API pour vérifier le statut du modèle"""
    return jsonify({
        'loaded': config.loaded,
        'loading': config.loading,
        'device': config.device
    })

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API pour générer une image"""
    try:
        data = request.json
        
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt requis'}), 400
        
        negative_prompt = data.get('negative_prompt', '')
        steps = min(max(int(data.get('steps', 20)), 10), 50)  # Limité entre 10 et 50
        guidance_scale = min(max(float(data.get('guidance_scale', 7.5)), 1.0), 20.0)
        width = min(max(int(data.get('width', 512)), 256), 768)  # Limité pour les ressources
        height = min(max(int(data.get('height', 512)), 256), 768)
        
        # Vérification que la taille est multiple de 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        image_b64, error = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        )
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{image_b64}",
            'prompt': prompt,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/examples')
def examples():
    """API pour obtenir des exemples de prompts"""
    example_prompts = [
        "A majestic dragon flying over a medieval castle at sunset, highly detailed, fantasy art",
        "Portrait of a cyberpunk warrior with neon lights, futuristic city background, digital art",
        "A serene Japanese garden with cherry blossoms, traditional architecture, peaceful atmosphere",
        "Astronaut exploring an alien planet with purple sky and strange rock formations, sci-fi",
        "A magical forest with glowing mushrooms and fairy lights, enchanted atmosphere",
        "Steampunk airship flying through clouds, brass and copper details, Victorian era",
        "Abstract geometric patterns in vibrant colors, modern art style",
        "A cozy library with floating books and magical atmosphere, warm lighting"
    ]
    
    return jsonify({'examples': example_prompts})

# Chargement du modèle au démarrage (en arrière-plan)
def init_model():
    """Initialise le modèle en arrière-plan"""
    thread = threading.Thread(target=load_model)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    # Initialisation du modèle
    init_model()
    
    # Démarrage du serveur
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
