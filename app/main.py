import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # For Vaja9

from app import (
    service_image,  # image service router
    service_main,  # main service router
    service_nlp,  # NLP service router
    service_nomadml,  # NomadML service
)

app = FastAPI(
    title="aiforthai-line-chatbot",
    description="AIFORTHAI LINE CHATBOT WORKSHOP",
    version="1.0.0",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(service_main.router)
app.include_router(service_nlp.router)
app.include_router(service_image.router)
app.include_router(service_nomadml.router)

# Save static files at the /static endpoint
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static/", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return "AIFORTHAI LINE CHATBOT WORKSHOP"
