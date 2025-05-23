from flask import Flask, render_template, request, jsonify
import requests
import base64
import io
import os
import time
import json
from datetime import datetime
import threading
from PIL import Image

app = Flask(__name__)

# Configuration globale
class Config:
    def __init__(self):
        self.ready = True
        self.apis = {
            'pollinations': 'https://image.pollinations.ai/prompt/',
            'picsum': 'https://picsum.photos/',
            'replicate': None,  # Nécessite une clé API
            'huggingface': 'https://api-inference.huggingface.co/models/'
        }
        self.hf_token = os.environ.get('HUGGINGFACE_TOKEN', '')
        self.current_api = 'pollinations'  # API par défaut
        
config = Config()

def generate_with_pollinations(prompt, width=512, height=512):
    """Génère une image avec Pollinations AI (gratuit, sans clé)"""
    try:
        # Nettoyage et optimisation du prompt
        clean_prompt = prompt.replace(' ', '%20').replace(',', '%2C')
        
        # URL de l'API Pollinations
        url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width={width}&height={height}&model=flux&enhance=true&nologo=true"
        
        print(f"🎨 Génération avec Pollinations: {prompt}")
        
        # Requête avec timeout
        response = requests.get(url, timeout=60, stream=True)
        
        if response.status_code == 200:
            # Conversion en image PIL
            image = Image.open(io.BytesIO(response.content))
            return image, None
        else:
            return None, f"Erreur API Pollinations: {response.status_code}"
            
    except Exception as e:
        print(f"Erreur Pollinations: {e}")
        return None, str(e)

def generate_with_huggingface(prompt, negative_prompt="", model="stabilityai/stable-diffusion-2-1"):
    """Génère une image avec Hugging Face Inference API"""
    if not config.hf_token:
        return None, "Token Hugging Face requis"
    
    try:
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {config.hf_token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512
            }
        }
        
        print(f"🤗 Génération avec Hugging Face: {prompt}")
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image, None
        elif response.status_code == 503:
            return None, "Modèle en cours de chargement, réessayez dans 30s"
        else:
            return None, f"Erreur API HF: {response.status_code}"
            
    except Exception as e:
        print(f"Erreur Hugging Face: {e}")
        return None, str(e)

def generate_with_dezgo(prompt, negative_prompt="", width=512, height=512):
    """Génère une image avec DezGo API (gratuit avec limite)"""
    try:
        API_URL = "https://api.dezgo.com/text2image"
        
        data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt or "blurry, bad quality, distorted",
            'model': 'epic_realism',
            'width': width,
            'height': height,
            'guidance': 7.5,
            'steps': 25,
            'sampler': 'dpmpp_2m'
        }
        
        print(f"🎯 Génération avec DezGo: {prompt}")
        
        response = requests.post(API_URL, data=data, timeout=90)
        
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image, None
        else:
            return None, f"Erreur API DezGo: {response.status_code}"
            
    except Exception as e:
        print(f"Erreur DezGo: {e}")
        return None, str(e)

def generate_with_craiyon(prompt):
    """Génère une image avec Craiyon API (ex-DALL-E mini)"""
    try:
        API_URL = "https://api.craiyon.com/v3"
        
        payload = {
            "prompt": prompt,
            "model": "art",
            "negative_prompt": "blurry, low quality",
            "version": "35s5hfwn9n78gb06"
        }
        
        print(f"🖍️ Génération avec Craiyon: {prompt}")
        
        response = requests.post(API_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            if 'images' in data and data['images']:
                # Craiyon retourne des images en base64
                img_data = base64.b64decode(data['images'][0])
                image = Image.open(io.BytesIO(img_data))
                return image, None
        
        return None, "Erreur génération Craiyon"
            
    except Exception as e:
        print(f"Erreur Craiyon: {e}")
        return None, str(e)

def enhance_prompt(prompt):
    """Améliore automatiquement le prompt pour de meilleurs résultats"""
    # Mots-clés pour améliorer la qualité
    quality_keywords = ", highly detailed, professional, 8k, masterpiece, photorealistic"
    
    # Si le prompt est court, on l'enrichit
    if len(prompt.split()) < 10:
        enhanced = f"{prompt}, {quality_keywords.strip(', ')}"
    else:
        enhanced = f"{prompt}{quality_keywords}"
    
    return enhanced

def generate_image(prompt, negative_prompt="", steps=20, guidance_scale=7.5, width=512, height=512, api_choice="auto"):
    """Génère une image avec gestion multi-API et fallback"""
    
    # Amélioration automatique du prompt
    enhanced_prompt = enhance_prompt(prompt)
    
    # Prompt négatif par défaut si vide
    if not negative_prompt.strip():
        negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    
    print(f"🚀 Génération: '{enhanced_prompt[:50]}...'")
    
    # Liste des APIs à essayer selon le choix
    if api_choice == "auto":
        apis_to_try = [
            ('pollinations', lambda: generate_with_pollinations(enhanced_prompt, width, height)),
            ('dezgo', lambda: generate_with_dezgo(enhanced_prompt, negative_prompt, width, height)),
            ('huggingface', lambda: generate_with_huggingface(enhanced_prompt, negative_prompt)),
            ('craiyon', lambda: generate_with_craiyon(enhanced_prompt))
        ]
    elif api_choice == "pollinations":
        apis_to_try = [('pollinations', lambda: generate_with_pollinations(enhanced_prompt, width, height))]
    elif api_choice == "huggingface" and config.hf_token:
        apis_to_try = [('huggingface', lambda: generate_with_huggingface(enhanced_prompt, negative_prompt))]
    else:
        apis_to_try = [('pollinations', lambda: generate_with_pollinations(enhanced_prompt, width, height))]
    
    # Essai des APIs dans l'ordre
    last_error = None
    for api_name, api_func in apis_to_try:
        try:
            print(f"⚡ Tentative avec {api_name}...")
            image, error = api_func()
            
            if image and not error:
                print(f"✅ Succès avec {api_name}!")
                config.current_api = api_name
                return image, None, api_name
            else:
                print(f"❌ Échec {api_name}: {error}")
                last_error = error
                
        except Exception as e:
            print(f"💥 Exception {api_name}: {e}")
            last_error = str(e)
            continue
    
    return None, last_error or "Toutes les APIs ont échoué", "none"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'loaded': True,
        'loading': False,
        'device': 'Cloud APIs',
        'current_api': config.current_api,
        'available_apis': {
            'pollinations': True,
            'dezgo': True,
            'huggingface': bool(config.hf_token),
            'craiyon': True
        }
    })

@app.route('/api/generate', methods=['POST'])
def api_generate():
    try:
        data = request.json
        
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt requis'}), 400
        
        if len(prompt) < 3:
            return jsonify({'error': 'Prompt trop court (minimum 3 caractères)'}), 400
        
        negative_prompt = data.get('negative_prompt', '')
        steps = min(max(int(data.get('steps', 25)), 10), 50)
        guidance_scale = min(max(float(data.get('guidance_scale', 7.5)), 1.0), 20.0)
        width = min(max(int(data.get('width', 512)), 256), 1024)
        height = min(max(int(data.get('height', 512)), 256), 1024)
        api_choice = data.get('api', 'auto')
        
        # Dimensions multiples de 64 pour certaines APIs
        width = (width // 64) * 64
        height = (height // 64) * 64
        
        print(f"📝 Requête reçue: {prompt[:30]}...")
        
        image, error, used_api = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            api_choice=api_choice
        )
        
        if error:
            return jsonify({
                'error': f'Erreur génération: {error}',
                'tried_apis': used_api
            }), 500
        
        # Conversion en base64
        buffer = io.BytesIO()
        # Optimisation de l'image
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'prompt': prompt,
            'enhanced_prompt': enhance_prompt(prompt),
            'api_used': used_api,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"💥 Erreur serveur: {e}")
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/examples')
def examples():
    example_prompts = [
        "A majestic dragon flying over a medieval castle at golden hour",
        "Portrait of a cyberpunk warrior with neon-lit city background",
        "Beautiful serene Japanese garden with cherry blossoms and koi pond",
        "Futuristic spaceship flying through a colorful nebula in deep space",
        "Enchanted magical forest with glowing mushrooms and fireflies",
        "Vintage steampunk airship with brass gears floating in cloudy sky",
        "Abstract fluid art with vibrant flowing colors and gold accents",
        "Majestic snow-capped mountain reflected in crystal clear lake",
        "Gothic cathedral interior with stained glass windows and divine light",
        "Underwater coral reef scene with tropical fish and sea creatures"
    ]
    
    return jsonify({
        'examples': example_prompts,
        'tips': [
            "Soyez descriptif et précis dans vos prompts",
            "Utilisez des mots-clés artistiques comme 'masterpiece', 'detailed'",
            "Spécifiez le style: 'photorealistic', 'digital art', 'oil painting'",
            "Mentionnez l'éclairage: 'golden hour', 'dramatic lighting'",
            "Ajoutez des détails techniques: '8K', 'sharp focus', 'professional'"
        ]
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'apis_status': 'operational'
    })

if __name__ == '__main__':
    print("🚀 Démarrage du Générateur d'Images AI Pro")
    print("🎨 APIs disponibles: Pollinations, DezGo, Craiyon" + (", Hugging Face" if config.hf_token else ""))
    print("✨ Mode: Production multi-API avec fallback intelligent")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
