#!/usr/bin/env python3
"""
Générateur d'Images AI Ultra-Réalistes - Version Production Améliorée
Support multiple: Stable Diffusion, FLUX, OpenAI DALL-E
Optimisé pour Render.com avec fallbacks intelligents
"""
 
import os
import io
import base64
import json
import uuid
import logging
import asyncio
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

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

class AdvancedImageGenerator:
    """Générateur d'images multi-modèles avec fallbacks intelligents"""
    
    def __init__(self):
        self.device = "cpu"
        self.available_models = []
        self.primary_model = None
        self.model_loaded = False
        self.is_loading = False
        
        # Configuration OpenAI
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.openai_available = bool(self.openai_api_key)
        
        # Configuration Replicate (pour FLUX et autres modèles avancés)
        self.replicate_api_key = os.environ.get('REPLICATE_API_TOKEN')
        self.replicate_available = bool(self.replicate_api_key)
        
        # Configuration Hugging Face
        self.hf_token = os.environ.get('HF_TOKEN')
        
        logger.info("Initialisation du générateur multi-modèles")
        self._check_available_services()
        
    def _check_available_services(self):
        """Vérifie les services disponibles"""
        services = []
        
        if self.openai_available:
            services.append("OpenAI DALL-E")
            
        if self.replicate_available:
            services.append("Replicate (FLUX, SDXL)")
            
        # Tentative de chargement des modèles locaux
        try:
            import torch
            if torch.cuda.is_available():
                services.append("Local GPU (Stable Diffusion)")
                self.device = "cuda"
            else:
                services.append("Local CPU (Stable Diffusion - Limité)")
        except ImportError:
            logger.warning("PyTorch non disponible")
            
        self.available_services = services
        logger.info(f"Services disponibles: {', '.join(services)}")
        
    def load_local_model(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        """Charge un modèle local avec optimisations avancées"""
        if self.model_loaded or self.is_loading:
            return
            
        self.is_loading = True
        logger.info(f"Chargement du modèle local: {model_name}")
        
        try:
            import torch
            from diffusers import (
                DiffusionPipeline, 
                AutoPipelineForText2Image,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler
            )
            
            # Sélection automatique du meilleur modèle disponible
            if self.device == "cuda":
                # GPU: utiliser SDXL ou FLUX
                try:
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16"
                    ).to("cuda")
                    
                    # Optimisations GPU
                    self.pipeline.enable_model_cpu_offload()
                    self.pipeline.enable_vae_slicing()
                    self.pipeline.enable_attention_slicing(1)
                    
                except Exception as e:
                    logger.warning(f"SDXL non disponible, fallback vers SD 1.5: {e}")
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to("cuda")
            else:
                # CPU: modèle optimisé
                self.pipeline = DiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                # Optimisations CPU
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_sequential_cpu_offload()
            
            # Scheduler optimisé
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.model_loaded = True
            self.primary_model = "local"
            logger.info("Modèle local chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle local: {e}")
            self.model_loaded = False
        finally:
            self.is_loading = False
    
    async def generate_with_openai(self, prompt, **kwargs):
        """Génération avec OpenAI DALL-E"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Optimisation du prompt pour DALL-E
            enhanced_prompt = self._enhance_prompt_for_dalle(prompt)
            
            # Paramètres DALL-E
            size = f"{kwargs.get('width', 1024)}x{kwargs.get('height', 1024)}"
            if size not in ["256x256", "512x512", "1024x1024"]:
                size = "1024x1024"
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt[:4000],  # Limite DALL-E
                size=size,
                quality="hd",
                n=1
            )
            
            # Télécharger l'image
            image_url = response.data[0].url
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            
            image = Image.open(io.BytesIO(img_response.content))
            return self._post_process_image(image, **kwargs)
            
        except Exception as e:
            logger.error(f"Erreur OpenAI: {e}")
            raise
    
    async def generate_with_replicate(self, prompt, **kwargs):
        """Génération avec Replicate (FLUX, SDXL, etc.)"""
        try:
            import replicate
            
            # Sélection du modèle selon la demande
            model_type = kwargs.get('model_type', 'flux')
            
            if model_type == 'flux':
                model = "black-forest-labs/flux-schnell"
                inputs = {
                    "prompt": self._enhance_prompt_for_flux(prompt),
                    "num_outputs": 1,
                    "aspect_ratio": f"{kwargs.get('width', 1024)}:{kwargs.get('height', 1024)}",
                    "output_format": "png",
                    "output_quality": 90
                }
            else:
                model = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
                inputs = {
                    "prompt": self._enhance_prompt_for_sdxl(prompt),
                    "negative_prompt": kwargs.get('negative_prompt', 'low quality, blurry'),
                    "width": kwargs.get('width', 1024),
                    "height": kwargs.get('height', 1024),
                    "num_inference_steps": kwargs.get('num_inference_steps', 30),
                    "guidance_scale": kwargs.get('guidance_scale', 7.5),
                    "seed": kwargs.get('seed')
                }
            
            output = replicate.run(model, input=inputs)
            
            # Télécharger l'image
            if isinstance(output, list):
                image_url = output[0]
            else:
                image_url = output
                
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            
            image = Image.open(io.BytesIO(img_response.content))
            return self._post_process_image(image, **kwargs)
            
        except Exception as e:
            logger.error(f"Erreur Replicate: {e}")
            raise
    
    def generate_with_local(self, prompt, **kwargs):
        """Génération avec modèle local"""
        if not self.model_loaded:
            self.load_local_model()
            
        if not self.model_loaded:
            raise Exception("Modèle local non disponible")
        
        try:
            import torch
            
            # Prompt amélioré
            enhanced_prompt = self._enhance_prompt_for_local(prompt)
            negative_prompt = kwargs.get('negative_prompt', 'low quality, blurry, distorted, deformed')
            
            # Paramètres optimisés
            params = {
                'prompt': enhanced_prompt,
                'negative_prompt': negative_prompt,
                'width': min(kwargs.get('width', 512), 1024 if self.device == "cuda" else 512),
                'height': min(kwargs.get('height', 512), 1024 if self.device == "cuda" else 512),
                'num_inference_steps': min(kwargs.get('num_inference_steps', 30), 50),
                'guidance_scale': kwargs.get('guidance_scale', 7.5),
                'eta': 0.0
            }
            
            # Générateur pour reproductibilité
            if kwargs.get('seed'):
                generator = torch.Generator(device=self.device).manual_seed(kwargs['seed'])
                params['generator'] = generator
            
            logger.info(f"Génération locale: {prompt[:50]}...")
            
            with torch.no_grad():
                result = self.pipeline(**params)
            
            return self._post_process_image(result.images[0], **kwargs)
            
        except Exception as e:
            logger.error(f"Erreur génération locale: {e}")
            raise
    
    def _enhance_prompt_for_dalle(self, prompt):
        """Optimise le prompt pour DALL-E"""
        return f"{prompt}, photorealistic, high quality, detailed, professional photography"
    
    def _enhance_prompt_for_flux(self, prompt):
        """Optimise le prompt pour FLUX"""
        return f"{prompt}, masterpiece, best quality, highly detailed, sharp focus, professional"
    
    def _enhance_prompt_for_sdxl(self, prompt):
        """Optimise le prompt pour SDXL"""
        return f"{prompt}, masterpiece, best quality, ultra detailed, 8k, photorealistic"
    
    def _enhance_prompt_for_local(self, prompt):
        """Optimise le prompt pour modèles locaux"""
        return f"{prompt}, high quality, detailed, masterpiece, best quality"
    
    def _post_process_image(self, image, **kwargs):
        """Post-traitement avancé de l'image"""
        try:
            # Redimensionnement si nécessaire
            target_width = kwargs.get('width', image.width)
            target_height = kwargs.get('height', image.height)
            
            if image.size != (target_width, target_height):
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Améliorations optionnelles
            enhance_level = kwargs.get('enhance_level', 'medium')
            
            if enhance_level in ['medium', 'high']:
                # Amélioration de la netteté
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
                
                # Amélioration du contraste
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.05)
                
            if enhance_level == 'high':
                # Amélioration des couleurs
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.1)
                
                # Réduction légère du bruit
                image = image.filter(ImageFilter.SMOOTH_MORE)
            
            return image
            
        except Exception as e:
            logger.warning(f"Erreur post-traitement: {e}")
            return image
    
    async def generate_image(self, prompt, **kwargs):
        """Génération d'image avec fallback intelligent"""
        prefer_service = kwargs.get('prefer_service', 'auto')
        
        # Logique de sélection automatique
        if prefer_service == 'auto':
            if self.replicate_available:
                prefer_service = 'replicate'
            elif self.openai_available:
                prefer_service = 'openai'
            else:
                prefer_service = 'local'
        
        # Tentatives avec fallbacks
        services_to_try = []
        
        if prefer_service == 'openai' and self.openai_available:
            services_to_try.extend(['openai', 'replicate', 'local'])
        elif prefer_service == 'replicate' and self.replicate_available:
            services_to_try.extend(['replicate', 'openai', 'local'])
        else:
            services_to_try.extend(['local', 'replicate', 'openai'])
        
        # Filtrer les services disponibles
        services_to_try = [s for s in services_to_try if 
                          (s == 'openai' and self.openai_available) or
                          (s == 'replicate' and self.replicate_available) or
                          s == 'local']
        
        last_error = None
        
        for service in services_to_try:
            try:
                logger.info(f"Tentative avec {service}")
                
                if service == 'openai':
                    return await self.generate_with_openai(prompt, **kwargs)
                elif service == 'replicate':
                    return await self.generate_with_replicate(prompt, **kwargs)
                else:  # local
                    return self.generate_with_local(prompt, **kwargs)
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Échec avec {service}: {e}")
                continue
        
        # Si tous les services échouent
        raise Exception(f"Tous les services ont échoué. Dernière erreur: {last_error}")

# Instance globale du générateur
image_generator = AdvancedImageGenerator()

# Template HTML amélioré
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Générateur d'Images AI Pro - Multi-Modèles</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container {
            max-width: 1200px; margin: 0 auto;
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
        .header p { opacity: 0.9; font-size: 1.1em; }
        .services-status {
            display: flex; justify-content: center; gap: 15px; margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .service-badge {
            padding: 5px 12px; border-radius: 15px; font-size: 12px;
            font-weight: 600; text-transform: uppercase;
        }
        .service-badge.available { background: #d4edda; color: #155724; }
        .service-badge.unavailable { background: #f8d7da; color: #721c24; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #2c3e50; }
        .form-group input, .form-group textarea, .form-group select {
            width: 100%; padding: 12px; border: 2px solid #e1e8ed;
            border-radius: 8px; font-size: 14px; transition: all 0.3s ease;
        }
        .form-group input:focus, .form-group textarea:focus, .form-group select:focus {
            outline: none; border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }
        .model-selector {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px; margin-bottom: 20px;
        }
        .model-option {
            padding: 15px; border: 2px solid #e1e8ed; border-radius: 10px;
            text-align: center; cursor: pointer; transition: all 0.3s ease;
        }
        .model-option:hover { border-color: #3498db; background: #f8f9fa; }
        .model-option.selected { border-color: #3498db; background: #e3f2fd; }
        .model-option .model-name { font-weight: 600; margin-bottom: 5px; }
        .model-option .model-desc { font-size: 12px; color: #666; }
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
            padding: 30px; min-height: 400px;
            display: flex; align-items: center; justify-content: center;
        }
        .generated-image { max-width: 100%; max-height: 600px; border-radius: 10px; }
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
        .advanced-options { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px; }
        .advanced-toggle { cursor: pointer; color: #3498db; font-weight: 600; }
        .advanced-content { display: none; margin-top: 15px; }
        .advanced-content.show { display: block; }
        @media (max-width: 768px) {
            .form-row, .form-row-3 { grid-template-columns: 1fr; }
            .header h1 { font-size: 2em; }
            .model-selector { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Générateur d'Images AI Pro</h1>
            <p>Multi-Modèles: DALL-E, FLUX, Stable Diffusion</p>
        </div>

        <div class="services-status" id="servicesStatus">
            <!-- Status sera injecté par JavaScript -->
        </div>

        <div id="status" class="status" style="display: none;"></div>

        <form id="generateForm">
            <div class="form-group">
                <label for="prompt">Description de l'image ✨</label>
                <textarea id="prompt" rows="3" placeholder="Décrivez l'image que vous souhaitez générer..." required></textarea>
            </div>

            <div class="form-group">
                <label>Modèle préféré 🤖</label>
                <div class="model-selector">
                    <div class="model-option selected" data-service="auto">
                        <div class="model-name">Auto</div>
                        <div class="model-desc">Sélection automatique</div>
                    </div>
                    <div class="model-option" data-service="openai">
                        <div class="model-name">DALL-E 3</div>
                        <div class="model-desc">OpenAI (Premium)</div>
                    </div>
                    <div class="model-option" data-service="replicate">
                        <div class="model-name">FLUX</div>
                        <div class="model-desc">Dernière génération</div>
                    </div>
                    <div class="model-option" data-service="local">
                        <div class="model-name">Stable Diffusion</div>
                        <div class="model-desc">Local/Gratuit</div>
                    </div>
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label for="width">Largeur</label>
                    <select id="width">
                        <option value="512">512px</option>
                        <option value="768">768px</option>
                        <option value="1024" selected>1024px</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="height">Hauteur</label>
                    <select id="height">
                        <option value="512">512px</option>
                        <option value="768">768px</option>
                        <option value="1024" selected>1024px</option>
                    </select>
                </div>
            </div>

            <div class="advanced-options">
                <div class="advanced-toggle" onclick="toggleAdvanced()">
                    ⚙️ Options avancées
                </div>
                <div class="advanced-content" id="advancedContent">
                    <div class="form-group">
                        <label for="negativePrompt">Éléments à éviter</label>
                        <textarea id="negativePrompt" rows="2" placeholder="Ex: flou, basse qualité, déformé..."></textarea>
                    </div>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label for="steps">Étapes</label>
                            <select id="steps">
                                <option value="20">20 (Rapide)</option>
                                <option value="30" selected>30 (Standard)</option>
                                <option value="50">50 (Qualité)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="guidance">Guidance</label>
                            <select id="guidance">
                                <option value="5">5 (Créatif)</option>
                                <option value="7.5" selected>7.5 (Équilibré)</option>
                                <option value="10">10 (Précis)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="seed">Seed</label>
                            <input type="number" id="seed" placeholder="Aléatoire">
                        </div>
                    </div>
                </div>
            </div>

            <button type="submit" class="generate-btn" id="generateBtn">
                <span class="btn-text">Générer l'Image 🚀</span>
            </button>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px;">Génération en cours... Cela peut prendre 30s à 2 minutes</p>
        </div>

        <div class="image-container" id="imageContainer">
            <div>
                <div style="font-size: 48px; margin-bottom: 10px;">🎨</div>
                <div>Votre image générée apparaîtra ici</div>
            </div>
        </div>
    </div>

    <script>
        let selectedService = 'auto';

        function toggleAdvanced() {
            const content = document.getElementById('advancedContent');
            content.classList.toggle('show');
        }

        // Sélection du modèle
        document.querySelectorAll('.model-option').forEach(option => {
            option.addEventListener('click', () => {
                document.querySelectorAll('.model-option').forEach(o => o.classList.remove('selected'));
                option.classList.add('selected');
                selectedService = option.dataset.service;
            });
        });

        const form = document.getElementById('generateForm');
        const statusDiv = document.getElementById('status');
        const loadingDiv = document.getElementById('loading');
        const imageContainer = document.getElementById('imageContainer');
        const generateBtn = document.getElementById('generateBtn');

        function showStatus(message, type) {
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            statusDiv.style.display = 'block';
            setTimeout(() => statusDiv.style.display = 'none', 8000);
        }

        function updateServicesStatus(services) {
            const container = document.getElementById('servicesStatus');
            container.innerHTML = services.map(service => 
                `<div class="service-badge ${service.available ? 'available' : 'unavailable'}">
                    ${service.name}
                </div>`
            ).join('');
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
                guidance_scale: parseFloat(document.getElementById('guidance').value),
                seed: document.getElementById('seed').value ? parseInt(document.getElementById('seed').value) : null,
                prefer_service: selectedService
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
                    showStatus(`Image générée avec ${result.service_used || 'succès'}!`, 'success');
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
                updateServicesStatus(data.services || []);
                if (data.ready) {
                    showStatus('Services prêts!', 'success');
                } else {
                    showStatus('Initialisation en cours...', 'info');
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
    """API de génération d'images avec support multi-modèles"""
    try:
        data = request.get_json()
        
        # Validation
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt requis'}), 400
        
        if len(prompt) > 1000:
            return jsonify({'error': 'Prompt trop long (max 1000 caractères)'}), 400
        
        # Paramètres avec validation
        params = {
            'prompt': prompt,
            'negative_prompt': data.get('negative_prompt', ''),
            'width': max(256, min(int(data.get('width', 1024)), 2048)),
            'height': max(256, min(int(data.get('height', 1024)), 2048)),
            'num_inference_steps': max(10, min(int(data.get('num_inference_steps', 30)), 100)),
            'guidance_scale': max(1.0, min(float(data.get('guidance_scale', 7.5)), 20.0)),
            'seed': data.get('seed'),
            'prefer_service': data.get('prefer_service', 'auto'),
            'enhance_level': data.get('enhance_level', 'medium')
        }
        
        logger.info(f"Génération demandée: {prompt[:50]}... (Service: {params['prefer_service']})")
        
        # Génération asynchrone
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            image = loop.run_until_complete(image_generator.generate_image(**params))
            service_used = "Multi-modèle"  # À améliorer pour tracker le service utilisé
        finally:
            loop.close()
        
        # Conversion en base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True, quality=95)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info("Image générée avec succès")
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{image_base64}",
            'service_used': service_used,
            'parameters': params
        })
        
    except Exception as e:
        logger.error(f"Erreur génération: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Status détaillé de l'API"""
    services = []
    
    # Vérification OpenAI
    if image_generator.openai_available:
        services.append({'name': 'DALL-E 3', 'available': True})
    else:
        services.append({'name': 'DALL-E 3', 'available': False})
    
    # Vérification Replicate
    if image_generator.replicate_available:
        services.append({'name': 'FLUX/SDXL', 'available': True})
    else:
        services.append({'name': 'FLUX/SDXL', 'available': False})
    
    # Vérification Local
    services.append({
        'name': f'Local ({image_generator.device.upper()})', 
        'available': True
    })
    
    return jsonify({
        'status': 'online',
        'ready': True,
        'services': services,
        'available_services': image_generator.available_services,
        'device': image_generator.device,
        'model_loaded': image_generator.model_loaded,
        'environment': 'production',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    """Health check pour Render"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/models')
def available_models():
    """Liste des modèles disponibles"""
    models = []
    
    if image_generator.openai_available:
        models.append({
            'id': 'openai',
            'name': 'DALL-E 3',
            'description': 'OpenAI DALL-E 3 - Ultra haute qualité',
            'max_size': 1024,
            'features': ['Ultra réaliste', 'Très rapide', 'Premium']
        })
    
    if image_generator.replicate_available:
        models.extend([
            {
                'id': 'replicate-flux',
                'name': 'FLUX Schnell',
                'description': 'FLUX - Génération rapide et créative',
                'max_size': 1024,
                'features': ['Très créatif', 'Rapide', 'Open source']
            },
            {
                'id': 'replicate-sdxl',
                'name': 'Stable Diffusion XL',
                'description': 'SDXL - Équilibre qualité/vitesse',
                'max_size': 1024,
                'features': ['Équilibré', 'Polyvalent', 'Stable']
            }
        ])
    
    models.append({
        'id': 'local',
        'name': 'Stable Diffusion Local',
        'description': f'Modèle local sur {image_generator.device.upper()}',
        'max_size': 1024 if image_generator.device == 'cuda' else 512,
        'features': ['Gratuit', 'Privé', 'Personnalisable']
    })
    
    return jsonify({'models': models})

# Point d'entrée pour Gunicorn
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"Démarrage de l'application sur le port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
