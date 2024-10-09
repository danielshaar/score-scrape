from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from modal.functions import FunctionCall

from main import scrape_score

from modal import App, asgi_app

api_app = App("api")
web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@api_app.function()
@asgi_app()
def fastapi_app():
    return web_app


@web_app.post("/accept")
async def accept_video(video: UploadFile = File(...)):
    video_bytes = await video.read()

    call = scrape_score.spawn(video_bytes)
    print("this is the call:", call)
    return {"call_id": call.object_id}


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
        print("result:", result)
        return {"markdown": result}
    except TimeoutError:
        return JSONResponse({"status": "processing"}, status_code=202)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(web_app, host="0.0.0.0", port=8000)
