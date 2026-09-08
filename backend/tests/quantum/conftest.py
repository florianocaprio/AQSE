import sys
from pathlib import Path


USER_PIPELINE = Path(__file__).resolve().parents[2] / "app" / "quantum" / "user_pipeline"
sys.path.insert(0, str(USER_PIPELINE))
