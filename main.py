from flask import Flask, render_template, request, jsonify
import requests
import base64
import io
import os
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import json
from datetime import datetime
import threading

app = Flask(__name__)

# Configuration
class Config:
    def __init__(self):
        self.ready = True
        self.using_huggingface = False
        self.hf_token = os.environ.get('HUGGINGFACE_TOKEN', '')

config = Config()

# Générateur d'images procédurales en cas d'absence d'API
def generate_procedural_image(prompt, width=512, height=512):
    """Génère une image procédurale basée sur le prompt"""
    try:
        # Création d'une image de base
        img = Image.new('RGB', (width, height), color=(20, 25, 40))
        draw = ImageDraw.Draw(img)
        
        # Analyse du prompt pour déterminer le style
        prompt_lower = prompt.lower()
        
        # Couleurs basées sur les mots-clés
        colors = []
        if 'dragon' in prompt_lower or 'fire' in prompt_lower or 'red' in prompt_lower:
            colors = [(255, 100, 100), (255, 150, 50), (200, 50, 50)]
        elif 'ocean' in prompt_lower or 'blue' in prompt_lower or 'water' in prompt_lower:
            colors = [(50, 150, 255), (100, 200, 255), (0, 100, 200)]
        elif 'forest' in prompt_lower or 'green' in prompt_lower or 'nature' in prompt_lower:
            colors = [(50, 200, 100), (100, 255, 150), (30, 150, 50)]
        elif 'sunset' in prompt_lower or 'orange' in prompt_lower:
            colors = [(255, 150, 50), (255, 100, 100), (255, 200, 100)]
        elif 'space' in prompt_lower or 'star' in prompt_lower or 'galaxy' in prompt_lower:
            colors = [(100, 50, 255), (150, 100, 255), (50, 0, 100)]
        else:
            colors = [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for _ in range(3)]
        
        # Génération de formes géométriques
        for _ in range(random.randint(20, 50)):
            color = random.choice(colors)
            alpha = random.randint(30, 100)
            
            # Création d'une image temporaire avec transparence
            temp_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            shape_type = random.choice(['circle', 'rectangle', 'polygon'])
            
            if shape_type == 'circle':
                x, y = random.randint(0, width), random.randint(0, height)
                r = random.randint(10, 100)
                temp_draw.ellipse([x-r, y-r, x+r, y+r], fill=(*color, alpha))
            
            elif shape_type == 'rectangle':
                x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
                x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
                temp_draw.rectangle([x1, y1, x2, y2], fill=(*color, alpha))
            
            else:  # polygon
                points = []
                for _ in range(random.randint(3, 8)):
                    points.append((random.randint(0, width), random.randint(0, height)))
                temp_draw.polygon(points, fill=(*color, alpha))
            
            # Fusion avec l'image principale
            img = Image.alpha_composite(img.convert('RGBA'), temp_img).convert('RGB')
        
        # Application d'effets
        if 'blur' not in prompt_lower and 'sharp' not in prompt_lower:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        if 'bright' in prompt_lower:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.3)
        
        if 'contrast' in prompt_lower:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
        
        # Ajout de texte artistique
        try:
            font_size = max(20, min(width, height) // 20)
            font = ImageFont.load_default()
            
            # Titre basé sur le prompt
            title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (width - text_width) // 2
            text_y = height - 40
            
            # Ombre du texte
            draw.text((text_x + 2, text_y + 2), title, fill=(0, 0, 0, 128), font=font)
            draw.text((text_x, text_y), title, fill=(255, 255, 255, 200), font=font)
        except:
            pass
        
        return img
    
    except Exception as e:
        # Image d'erreur simple
        img = Image.new('RGB', (width, height), color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        draw.text((width//2-50, height//2), "Erreur", fill=(255, 255, 255))
        return img

def try_huggingface_api(prompt, negative_prompt="", width=512, height=512):
    """Essaie d'utiliser l'API Hugging Face si disponible"""
    if not config.hf_token:
        return None
    
    try:
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Authorization": f"Bearer {config.hf_token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image
        else:
            print(f"Erreur API HF: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Erreur HuggingFace: {e}")
        return None

def generate_image(prompt, negative_prompt="", steps=20, guidance_scale=7.5, width=512, height=512):
    """Génère une image avec fallback sur génération procédurale"""
    try:
        # Tentative avec HuggingFace d'abord
        if config.hf_token:
            hf_image = try_huggingface_api(prompt, negative_prompt, width, height)
            if hf_image:
                config.using_huggingface = True
                return hf_image, None
        
        # Fallback sur génération procédurale
        config.using_huggingface = False
        image = generate_procedural_image(prompt, width, height)
        return image, None
        
    except Exception as e:
        return None, f"Erreur lors de la génération: {str(e)}"

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """API pour vérifier le statut"""
    return jsonify({
        'loaded': True,
        'loading': False,
        'device': 'CPU',
        'method': 'HuggingFace API' if config.using_huggingface else 'Génération Procédurale',
        'hf_available': bool(config.hf_token)
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
        steps = min(max(int(data.get('steps', 20)), 10), 50)
        guidance_scale = min(max(float(data.get('guidance_scale', 7.5)), 1.0), 20.0)
        width = min(max(int(data.get('width', 512)), 256), 1024)
        height = min(max(int(data.get('height', 512)), 256), 1024)
        
        # Ajustement des dimensions pour être multiples de 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        image, error = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        )
        
        if error:
            return jsonify({'error': error}), 500
        
        # Conversion en base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{img_base64}",
            'prompt': prompt,
            'method': 'HuggingFace API' if config.using_huggingface else 'Génération Procédurale',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/examples')
def examples():
    """API pour obtenir des exemples de prompts"""
    example_prompts = [
        "A majestic dragon flying over a medieval castle at sunset",
        "Cyberpunk city with neon lights and flying cars",
        "Beautiful landscape with mountains and a lake",
        "Abstract geometric art with vibrant colors",
        "Futuristic robot in a sci-fi laboratory",
        "Magical forest with glowing mushrooms",
        "Ocean waves crashing on a rocky shore",
        "Space scene with planets and stars",
        "Vintage car on an old country road",
        "Modern architecture building at night"
    ]
    
    return jsonify({'examples': example_prompts})

@app.route('/health')
def health():
    """Health check pour Render"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("🚀 Démarrage du générateur d'images AI...")
    print(f"🔧 Token HuggingFace: {'✅ Configuré' if config.hf_token else '❌ Non configuré (mode procédural)'}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
