from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lexis.api.routes import router

app = FastAPI(title="LEXIS RAG API", version="1.0.0")

# Permit frontends to connect without CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from lexis.retrieval.engine import RetrievalEngine

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    # Automatically initialize empty database schemas when the server starts
    engine = RetrievalEngine()
    await engine.qdrant.initialize_collections()
    await engine.es.initialize_index()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
