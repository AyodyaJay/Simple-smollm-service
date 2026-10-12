from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

print("Loading open model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

app = FastAPI(
    title="Simple SmolLM2 Open Model Service",
    description="A simple custom REST API wrapping an open-weight SmolLM2 model.",
    version="1.0.0"
)

class GenerateRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "service": "Simple SmolLM2 Open Model Service",
        "model": MODEL_NAME,
        "device": device,
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": device
    }

@app.post("/generate")
def generate(request: GenerateRequest):
    messages = [{"role": "user", "content": request.text}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return {
        "model": MODEL_NAME,
        "response": response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
