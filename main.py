import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os

def download_model():
    from diffusers import AutoPipelineForText2Image
    import torch
    
    AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
image = (modal.Image.debian_slim()
         .pip_install("fastapi[standard]", "transformers", "accelerate", "diffusers", "requests")
         .run_function(download_model))

app = modal.App("Stable-Diffusion", image=image)

#within our docker container
@app.cls(
    image=image,
    gpu="A10G",
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("api-key")]
)

class Model:
    
    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch
        
        self.pipe = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sdxl-turbo",
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
        
        self.pipe.to("cuda")
        self.API_KEY = os.environ["API_KEY"]
        
    #create our modal endpoint (API endpoint for frontend to call)
    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str=Query(..., description="Prompt for image generation")):
        
        api_key = request.headers.get("X-API-KEY")
        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )
        
        image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0)
        
        # create a buffer to save the image to as it's more efficient than saving to your machine
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        
        return Response(content=buffer.getvalue(), media_type="image/jpeg")
    @modal.web_endpoint()
    def health(self):
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
    
    
@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[modal.Secret.from_name("api-key")]
)
def update_keep_warm():
    health_url = "https://ethnwang--stable-diffusion-model-generate.modal.run"
    generate_url = "https://ethnwang--stable-diffusion-model-generate.modal.run"
    
    health_response = requests.get(health_url)
    print(f"Health check at: {health_response.json()['timestamp']}")
    
    headers = {"X-Api_Key": os.environ["API_KEY"]}
    generate_response = requests.get(generate_url, header=headers)
    print(f"Generate endpoint tested successfully at: {datetime.now(timezone.utc).isoformat()}")