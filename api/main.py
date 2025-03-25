import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from .generate import get_model, generate_text

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="