# Legal Document Analyzer

This project is a legal document analyzer that uses AI to process and analyze legal documents.

## Server Requirements

### Python Version
- Python 3.13

### Environment Variables
The following environment variables need to be set. It is recommended to use a `.env` file for local development. An example file `.env.example` is provided.

To create your own `.env` file from the example, you can use the following command in your terminal:

```bash
cp .env.example .env
```

Then, edit the `.env` file with your specific configurations.

- `OPENAI_API_KEY`: Your API key for OpenAI.
- `TESSERACT_CMD`: Path to the Tesseract executable (e.g., `/usr/bin/tesseract` or `C:\\Program Files\\Tesseract-OCR\\tesseract.exe` on Windows).
- `GOOGLE_CLOUD_VISION_API_KEY`: Your API key for Google Cloud Vision.
- `DATABASE_URL`: The connection string for your PostgreSQL database (e.g., `postgresql+psycopg://postgres:YOUR_DB_PASSWORD@localhost:5432/legal`).
- `JWT_SECRET_KEY`: A secret key for JWT authentication.

## Deployment

### Local Development (without Docker)
1.  **Install Python 3.13.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your PostgreSQL database.**
5.  **Set up your environment variables:**
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Then, edit the `.env` file with your local development credentials and paths.
    Ensure your `.env` file for local development includes:
    - `OPENAI_API_KEY="your_openai_api_key"`
    - `TESSERACT_CMD="/path/to/tesseract"` (update if different for your local OS, e.g., `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`)
    - `GOOGLE_CLOUD_VISION_API_KEY="your_google_cloud_vision_key"`
    - `DATABASE_URL="postgresql+psycopg://postgres:YOUR_DB_PASSWORD@localhost:5432/legal"` (adjust to your local PostgreSQL setup)
    - `JWT_SECRET_KEY="your_strong_jwt_secret"`
6.  **Run the FastAPI application:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
7.  **To run the Streamlit app (optional, for UI):**
    ```bash
    streamlit run streamlit_app.py
    ```

### Production (without Docker)
1.  **Ensure Python 3.13 is installed** on your server.
2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install gunicorn  # Or another production-grade WSGI server
    ```
4.  **Set up your PostgreSQL database** for production.
5.  **Set the required environment variables securely** on your server. For your own server, you can manage these variables by:
    *   Creating an environment file (e.g., `.env`). Ensure this file is **not** committed to your version control system (e.g., add it to `.gitignore`). Load these variables into your application's environment when it starts.
    *   Setting them as system-wide environment variables.    The `.env.example` file shows which variables are needed.
    - `OPENAI_API_KEY="your_production_openai_api_key"`
    - `TESSERACT_CMD="/usr/bin/tesseract"` (ensure this path is correct for your server)
    - `GOOGLE_CLOUD_VISION_API_KEY="your_production_google_cloud_vision_key"`
    - `DATABASE_URL="postgresql+psycopg://USER:PASSWORD@PRODUCTION_DB_HOST:PORT/DATABASE"`
    - `JWT_SECRET_KEY="your_very_strong_production_jwt_secret"`
6.  **Run the FastAPI application using a production WSGI server (e.g., Gunicorn):**
    ```bash
    gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    ```
    Adjust the number of workers (`--workers`) based on your server's CPU cores.

