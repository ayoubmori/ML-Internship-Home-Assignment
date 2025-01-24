import uvicorn
from data_ml_assignment.api.server import server

if __name__ == "__main__":
    uvicorn.run(server, host="localhost", port=9000, log_level="info")