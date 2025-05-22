from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
import requests
import io
import base64
import os
import uuid
from datetime import datetime
import logging
from PIL import Image
import json

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration optimisée pour Render
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = 'generated_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configuration des caches pour optimiser l'espace
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/hf_cache'

class AIImageGenerator:
    def __init__(self):
        self.api_mode = True  # Mode API par défaut pour Render gratuit
        self.hf_token = os.environ.get('HF_TOKEN', '')  # Token Hugging Face optionnel
        
    def generate_image_api(self, prompt, negative_prompt="", width=512, height=512, 
                          num_inference_steps=20, guidance_scale=7.5):
        """Génération via l'API Hugging Face (recommandé pour Render gratuit)"""
        try:
            logger.info(f"Génération via API pour: {prompt}")
            
            # URL de l'API Hugging Face Inference
            api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
            
            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return image
            else:
                logger.error(f"Erreur API: {response.status_code} - {response.text}")
                # Fallback vers génération locale si API échoue
                return self.generate_image_local(prompt, negative_prompt, width, height, 
                                               num_inference_steps, guidance_scale)
                
        except Exception as e:
            logger.error(f"Erreur API: {str(e)}")
            # Fallback vers génération locale
            return self.generate_image_local(prompt, negative_prompt, width, height, 
                                           num_inference_steps, guidance_scale)
    
    def generate_image_local(self, prompt, negative_prompt="", width=256, height=256, 
                           num_inference_steps=10, guidance_scale=7.5):
        """Génération locale avec modèle léger (fallback)"""
        try:
            logger.info("Tentative de génération locale...")
            
            # Import conditionnel pour éviter les erreurs si pas assez de RAM
            from diffusers import StableDiffusionPipeline
            
            # Utilisation d'un modèle plus léger
            model_id = "dreamlike-art/dreamlike-diffusion-1.0"
            
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                cache_dir="/tmp/models"
            )
            
            pipe = pipe.to("cpu")
            
            # Génération avec paramètres réduits pour économiser la RAM
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=min(width, 256),  # Limite la taille
                height=min(height, 256),
                num_inference_steps=min(num_inference_steps, 10),
                guidance_scale=guidance_scale
            ).images[0]
            
            # Libération immédiate de la mémoire
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return image
            
        except Exception as e:
            logger.error(f"Erreur génération locale: {str(e)}")
            # Génération d'une image placeholder en cas d'échec
            return self.generate_placeholder_image(prompt)
    
    def generate_placeholder_image(self, prompt):
        """Génère une image placeholder en cas d'échec"""
        logger.info("Génération d'image placeholder")
        
        # Création d'une image simple avec le prompt
        img_width, img_height = 512, 512
        image = Image.new('RGB', (img_width, img_height), color='#667eea')
        
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Texte centré
            text = f"Image générée pour:\n{prompt[:50]}..."
            
            # Essaie d'utiliser une police par défaut
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Calcul de la position centrée
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            position = ((img_width - text_width) // 2, (img_height - text_height) // 2)
            draw.text(position, text, fill='white', font=font)
            
        except Exception as e:
            logger.error(f"Erreur création placeholder: {str(e)}")
        
        return image
    
    def generate_image(self, prompt, **kwargs):
        """Point d'entrée principal pour la génération"""
        if self.api_mode:
            return self.generate_image_api(prompt, **kwargs)
        else:
            return self.generate_image_local(prompt, **kwargs)

# Instance globale
generator = AIImageGenerator()

@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    """Endpoint pour générer une image"""
    try:
        data = request.get_json()
        
        # Validation des paramètres
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Le prompt ne peut pas être vide'}), 400
        
        # Amélioration automatique du prompt pour de meilleurs résultats
        enhanced_prompt = f"{prompt}, high quality, detailed, photorealistic, 4k"
        
        negative_prompt = data.get('negative_prompt', 'blurry, low quality, distorted')
        width = min(max(int(data.get('width', 512)), 256), 512)  # Limite pour Render
        height = min(max(int(data.get('height', 512)), 256), 512)
        steps = min(max(int(data.get('steps', 20)), 10), 25)  # Réduit pour performance
        guidance = min(max(float(data.get('guidance', 7.5)), 1.0), 15.0)
        seed = data.get('seed')
        
        logger.info(f"Génération demandée: {enhanced_prompt}")
        
        # Génération de l'image
        image = generator.generate_image(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        
        # Sauvegarde de l'image
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath, optimize=True, quality=85)  # Compression pour économiser l'espace
        
        # Conversion en base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG', optimize=True, quality=85)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{img_base64}",
            'filename': filename,
            'parameters': {
                'prompt': prompt,
                'enhanced_prompt': enhanced_prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'steps': steps,
                'guidance': guidance,
                'seed': seed
            },
            'generation_method': 'api' if generator.api_mode else 'local'
        })
        
    except Exception as e:
        logger.error(f"Erreur dans generate_image: {str(e)}")
        return jsonify({
            'error': f'Erreur lors de la génération: {str(e)}',
            'suggestion': 'Essayez avec un prompt plus simple ou réduisez les dimensions'
        }), 500

@app.route('/download/<filename>')
def download_image(filename):
    """Téléchargement d'une image générée"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'Fichier non trouvé'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Vérification de l'état du service"""
    return jsonify({
        'status': 'healthy',
        'mode': 'api' if generator.api_mode else 'local',
        'device': 'cpu',
        'memory_usage': get_memory_usage(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/toggle-mode', methods=['POST'])
def toggle_generation_mode():
    """Basculer entre API et génération locale"""
    generator.api_mode = not generator.api_mode
    return jsonify({
        'mode': 'api' if generator.api_mode else 'local',
        'message': f"Mode basculé vers: {'API' if generator.api_mode else 'Local'}"
    })

def get_memory_usage():
    """Obtient l'utilisation mémoire approximative"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024   # MB
        }
    except:
        return {'error': 'psutil not available'}

# Nettoyage automatique des fichiers anciens
@app.before_request
def cleanup_old_images():
    """Nettoie les images anciennes pour économiser l'espace"""
    try:
        import time
        current_time = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                # Supprime les fichiers de plus d'une heure
                if file_age > 3600:
                    os.remove(filepath)
                    logger.info(f"Image supprimée: {filename}")
    except Exception as e:
        logger.error(f"Erreur nettoyage: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
