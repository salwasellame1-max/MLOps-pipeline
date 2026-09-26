# Base image: an official, lightweight Python 3.11 image (already has Python installed)
FROM python:3.11-slim

# Everything from here happens inside the folder /app in the container
WORKDIR /app

# Copy ONLY the requirements file first (see explanation: Docker layer caching)
COPY requirements.txt .

# Install the Python libraries inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project (code, config) into the container
COPY src/ src/
COPY models/ models/

# Document that the API listens on port 8000 (informational, doesn't open it by itself)
EXPOSE 8000

# The command run when the container starts
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]