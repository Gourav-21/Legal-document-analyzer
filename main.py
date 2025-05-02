from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
# from database import Base, engine
# from routers import auth_router
from routers.labor_law_api import router as labor_law_router
from routers.document_api import router as document_router
from routers.letter_format_api import router as letter_format_router
# from auth import get_current_user
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create database tables
# Base.metadata.create_all(bind=engine)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# app.include_router(auth_router.router, prefix="/auth", tags=["authentication"])
app.include_router(
    labor_law_router,
    prefix="/api",
    tags=["labor_laws"],
    # dependencies=[Depends(get_current_user)]
)
app.include_router(
    document_router,
    prefix="/api",
    tags=["documents"],
    # dependencies=[Depends(get_current_user)]
)
app.include_router(
    letter_format_router,
    prefix="/api",
    tags=["letter_formats"],
    # dependencies=[Depends(get_current_user)]
)

@app.get("/")
async def root():
    return {"message": "Welcome to Legal Document API"}

