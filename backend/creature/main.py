"""ASGI entry point: `uvicorn creature.main:app`."""
import logging

from .api import create_app

logging.basicConfig(level=logging.INFO)
app = create_app()
