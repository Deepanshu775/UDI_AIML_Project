#!/usr/bin/env bash
set -e
cd backend
python3 -m venv venv || true
source venv/bin/activate
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
python sample_data/generate_samples.py
python train_models.py
cd ../frontend
npm install
npm run build
cd ../backend
python app.py
