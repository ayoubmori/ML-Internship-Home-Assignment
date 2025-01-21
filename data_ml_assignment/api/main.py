# main.py

import uvicorn
from data_ml_assignment.api.server import server  # Assuming server is an instance of FastAPI

if __name__ == "__main__":
    uvicorn.run(server, host="localhost", port=9000, log_level="info")