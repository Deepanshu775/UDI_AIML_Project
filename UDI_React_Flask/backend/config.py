from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
MOCK_MODE = os.getenv("UDI_MOCK_MODE", "false").lower() == "true"
PORT = int(os.getenv("FLASK_PORT", "5000"))
UPLOAD_DIR = BASE_DIR / "uploads"
SAMPLE_DIR = BASE_DIR / "sample_data"
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DOMAINS = {
    "company": {"label":"Company", "icon":"▣", "target":"Profit", "forecast_target":"Demand", "description":"Sales, demand, inventory, customers and profitability."},
    "hospital": {"label":"Hospital", "icon":"✚", "target":"Patients", "forecast_target":"Patients", "description":"Patient load, capacity, waiting time and staffing."},
    "college": {"label":"College", "icon":"⌂", "target":"Attendance", "forecast_target":"Attendance", "description":"Attendance, students, faculty and resource utilization."},
    "municipality": {"label":"Municipality", "icon":"◈", "target":"Waste_Collected", "forecast_target":"Complaints", "description":"Waste, complaints, service demand and zones."},
    "ngo": {"label":"NGO", "icon":"♡", "target":"Beneficiaries", "forecast_target":"Beneficiaries", "description":"Beneficiaries, donations, aid and volunteer demand."},
    "community": {"label":"Community", "icon":"⌁", "target":"Resource_Requests", "forecast_target":"Resource_Requests", "description":"Local requests, participation, shortages and needs."},
    "event_organizer": {"label":"Event Organizer", "icon":"★", "target":"Attendance", "forecast_target":"Attendance", "description":"Attendance, capacity, registrations and resources."},
}
