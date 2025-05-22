from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import requests
import io
import base64
import os
import uuid
from datetime import datetime
import logging
from PIL import Image, ImageDraw, ImageFont
import json
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = 'generated_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SimpleAIImageGenerator:
    def __init__(self):
        self.hf_token = os.environ.get('HF_TOKEN', '')
        self.api_urls = [
            "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
            "https://api-inference.huggingface.co/models/dreamlike-art/dreamlike-diffusion-1.0"
        ]
        
    def generate_image(self, prompt, negative_prompt="", width=512, height=512, **kwargs):
        """Génération d'image via API Hugging Face"""
        
        # Amélioration du prompt
        enhanced_prompt = f"{prompt}, high quality, detailed, masterpiece"
        if negative_prompt:
            negative_prompt += ", blurry, low quality, distorted, ugly"
        else:
            negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"
        
        # Tentative avec différentes APIs
        for api_url in self.api_urls:
            try:
                logger.info(f"Tentative avec: {api_url}")
                
                headers = {"Content-Type": "application/json"}
                if self.hf_token:
                    headers["Authorization"] = f"Bearer {self.hf_token}"
                
                # Payload simple pour éviter les erreurs
                payload = {
                    "inputs": enhanced_prompt
                }
                
                # Si le modèle supporte les paramètres avancés
                if "stable-diffusion-v1-5" in api_url or "stable-diffusion-2" in api_url:
                    payload["parameters"] = {
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": min(kwargs.get('steps', 20), 25),
                        "guidance_scale": kwargs.get('guidance', 7.5)
                    }
                
                response = requests.post(
                    api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=60
                )
                
                if response.status_code == 200:
                    try:
                        image = Image.open(io.BytesIO(response.content))
                        # Redimensionner si nécessaire
                        if image.size != (width, height):
                            image = image.resize((width, height), Image.Resampling.LANCZOS)
                        logger.info(f"Image générée avec succès via {api_url}")
                        return image
                    except Exception as e:
                        logger.error(f"Erreur traitement image: {str(e)}")
                        continue
                        
                elif response.status_code == 503:
                    logger.warning(f"Modèle en cours de chargement: {api_url}")
                    # Attendre un peu et continuer avec le prochain modèle
                    time.sleep(2)
                    continue
                else:
                    logger.warning(f"Erreur API {response.status_code}: {response.text}")
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout pour {api_url}")
                continue
            except Exception as e:
                logger.error(f"Erreur {api_url}: {str(e)}")
                continue
        
        # Si toutes les APIs échouent, générer une image placeholder
        logger.warning("Toutes les APIs ont échoué, génération d'un placeholder")
        return self.create_placeholder_image(prompt, width, height)
    
    def create_placeholder_image(self, prompt, width=512, height=512):
        """Crée une image placeholder avec le texte du prompt"""
        
        # Couleurs gradient
        colors = [
            (102, 126, 234),  # Bleu
            (118, 75, 162),   # Violet
            (240, 147, 251)   # Rose
        ]
        
        # Création de l'image avec gradient
        image = Image.new('RGB', (width, height), colors[0])
        
        try:
            draw = ImageDraw.Draw(image)
            
            # Créer un effet de gradient simple
            for y in range(height):
                ratio = y / height
                r = int(colors[0][0] + (colors[1][0] - colors[0][0]) * ratio)
                g = int(colors[0][1] + (colors[1][1] - colors[0][1]) * ratio)
                b = int(colors[0][2] + (colors[1][2] - colors[0][2]) * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Ajouter le texte
            text_lines = [
                "🎨 Image générée par IA",
                "",
                f"Prompt: {prompt[:40]}{'...' if len(prompt) > 40 else ''}",
                "",
                "Générateur AI - Render"
            ]
            
            # Police par défaut
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Calcul de la position du texte
            y_offset = height // 2 - (len(text_lines) * 30) // 2
            
            for i, line in enumerate(text_lines):
                if line:  # Skip empty lines
                    font = font_large if i == 0 else font_small
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                    x = (width - text_width) // 2
                    y = y_offset + i * 30
                    
                    # Ombre
                    draw.text((x+1, y+1), line, fill=(0, 0, 0, 128), font=font)
                    # Texte principal
                    draw.text((x, y), line, fill='white', font=font)
            
        except Exception as e:
            logger.error(f"Erreur création placeholder: {str(e)}")
        
        return image

# Instance globale
generator = SimpleAIImageGenerator()

@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    """Endpoint pour générer une image"""
    try:
        data = request.get_json()
        
        # Validation
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Le prompt ne peut pas être vide'}), 400
        
        if len(prompt) > 500:
            return jsonify({'error': 'Le prompt est trop long (max 500 caractères)'}), 400
        
        # Paramètres
        negative_prompt = data.get('negative_prompt', '')
        width = min(max(int(data.get('width', 512)), 256), 768)
        height = min(max(int(data.get('height', 512)), 256), 768)
        steps = min(max(int(data.get('steps', 20)), 10), 30)
        guidance = min(max(float(data.get('guidance', 7.5)), 1.0), 15.0)
        
        logger.info(f"Génération: {prompt[:100]}...")
        
        # Génération
        image = generator.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance
        )
        
        # Sauvegarde
        filename = f"ai_image_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Optimisation de l'image pour réduire la taille
        image.save(filepath, 'PNG', optimize=True)
        
        # Conversion base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True, quality=90)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{img_base64}",
            'filename': filename,
            'parameters': {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'steps': steps,
                'guidance': guidance
            },
            'info': 'Générée via API Hugging Face'
        })
        
    except Exception as e:
        logger.error(f"Erreur génération: {str(e)}")
        return jsonify({
            'error': f'Erreur: {str(e)}',
            'suggestion': 'Réessayez avec un prompt plus simple'
        }), 500

@app.route('/download/<filename>')
def download_image(filename):
    """Téléchargement d'image"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'Fichier introuvable'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Santé du service"""
    return jsonify({
        'status': 'healthy',
        'mode': 'api_only',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    })

@app.route('/models')
def list_models():
    """Liste des modèles disponibles"""
    models = []
    for url in generator.api_urls:
        model_name = url.split('/')[-1]
        models.append({
            'name': model_name,
            'url': url,
            'description': f'Modèle {model_name}'
        })
    
    return jsonify({'models': models})

# Nettoyage automatique des fichiers
def cleanup_old_files():
    """Nettoie les vieux fichiers"""
    try:
        current_time = time.time()
        cleaned = 0
        
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                age = current_time - os.path.getmtime(filepath)
                if age > 3600:  # 1 heure
                    os.remove(filepath)
                    cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Nettoyage: {cleaned} fichiers supprimés")
            
    except Exception as e:
        logger.error(f"Erreur nettoyage: {str(e)}")

# Nettoyage périodique
@app.before_request
def periodic_cleanup():
    """Nettoyage périodique"""
    if not hasattr(app, 'last_cleanup'):
        app.last_cleanup = 0
    
    current_time = time.time()
    if current_time - app.last_cleanup > 1800:  # 30 minutes
        cleanup_old_files()
        app.last_cleanup = current_time

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
