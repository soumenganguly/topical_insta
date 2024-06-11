# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Set the working directory.
WORKDIR /insta_topical

# Install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY . .

# Expose the port the app runs on.
EXPOSE 8000

# Run the application.
CMD ["python", "app.py"]
