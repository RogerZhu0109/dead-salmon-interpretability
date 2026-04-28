FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY . .

# Non-root user required by HF Spaces
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

CMD ["marimo", "run", "--include-code", "--host", "0.0.0.0", "--port", "7860", "notebooks/walkthrough.py"]
