import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import settings


from backend.train_service import (
    search_trains,
    search_by_city,
    resolve_station,
    get_train_route,
)

from backend.chat_service import chat

from langchain_core.messages import HumanMessage

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Preloading RAG embedding model & reranker on backend startup...")
    try:
        try:
            from backend.rag.retriever import get_embedding_model
            from backend.rag.reranker import get_reranker
        except ModuleNotFoundError:
            from rag.retriever import get_embedding_model
            from rag.reranker import get_reranker
        get_embedding_model()
        get_reranker()
        print("✅ RAG models preloaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning during RAG preloading: {e}")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global exception handler — catch any unhandled server-side error and return
# a clean JSON response (HTTP 200 so the frontend can always parse the body).
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"UNHANDLED EXCEPTION on {request.url}: {exc}")
    return JSONResponse(
        status_code=200,
        content={
            "message": "",
            "response": (
                "❌ An unexpected server error occurred. "
                "Please try again or rephrase your question."
            ),
            "thread_id": "unknown",
            "error": True,
        },
    )


class TrainSearchRequest(BaseModel):
    origin: str
    destination: str
    preference: str = "balanced"

class StationSearchRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/stations/search")
def station_search(
    request: StationSearchRequest,
):
    try:
        stations = resolve_station(request.query)
    except Exception as exc:
        print(f"Station search error: {exc}")
        return {
            "query": request.query,
            "count": 0,
            "stations": [],
            "error": "An error occurred while searching for stations.",
        }

    return {
        "query": request.query,
        "count": len(stations),
        "stations": stations,
    }


@app.post("/trains/search")
def train_search(
    request: TrainSearchRequest,
):
    origin = request.origin.strip()
    destination = request.destination.strip()

    if not origin or not destination:
        return {
            "origin": origin,
            "destination": destination,
            "count": 0,
            "trains": [],
            "error": "Origin and destination must not be empty.",
        }

    try:
        trains = search_by_city(
            origin=origin,
            destination=destination,
            preference=request.preference,
        )
    except Exception as exc:
        print(f"Train search error: {exc}")
        return {
            "origin": origin,
            "destination": destination,
            "count": 0,
            "trains": [],
            "error": "An error occurred while searching for trains. Please try again.",
        }

    return {
        "origin": origin,
        "destination": destination,
        "preference": request.preference,
        "count": len(trains.get("trains", [])),
        "trains": trains.get("trains", []),
        "recommended_train": trains.get("recommended_train"),
        "reason": trains.get("reason"),
        "ranking_preference": trains.get("ranking_preference"),
    }

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


@app.post("/chat")
def chat_endpoint(
    request: ChatRequest,
):
    start = time.perf_counter()

    print("\n========== CHAT REQUEST ==========")
    print("USER:", request.message)

    response = chat(
        message=request.message,
        thread_id=request.thread_id,
    )

    elapsed = time.perf_counter() - start
    print(f"RESPONSE TIME: {elapsed:.2f}s")

    return {
        "message": request.message,
        "response": response,
        "thread_id": request.thread_id,
    }

@app.get("/trains/{train_number}")
def train_details(
    train_number: str,
):
    try:
        route = get_train_route(train_number)
    except Exception as exc:
        print(f"Train details error: {exc}")
        return {
            "train_number": train_number,
            "stop_count": 0,
            "route": [],
            "error": "An error occurred while fetching train details.",
        }

    if not route:
        return {
            "train_number": train_number,
            "stop_count": 0,
            "route": [],
            "error": f"No route information found for train {train_number}.",
        }

    return {
        "train_number": train_number,
        "stop_count": len(route),
        "route": route,
    }