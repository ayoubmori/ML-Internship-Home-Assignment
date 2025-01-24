from fastapi import FastAPI

from data_ml_assignment.api.inference_route import inference_router
from data_ml_assignment.api.constants import APP_NAME, API_PREFIX
from fastapi.middleware.cors import CORSMiddleware

def server() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        docs_url=f"{API_PREFIX}/docs",
    )
    app.include_router(inference_router, prefix=API_PREFIX)
    
# Define the allowed origins
    origins = [
        "http://localhost:9000",  # Replace with the URL of your frontend or client
        "http://127.0.0.1:9000",
        "http://localhost:3000",  # Example: React frontend
    ]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # List of allowed origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all HTTP headers
    )
    return app
