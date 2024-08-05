# predict.py
import torch
from diffusers import FluxPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = FluxPipeline.from_pretrained("enhanceaiteam/kalpana", torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()

    def predict(
        self,
        prompt: str = Input(description="Prompt for the image generation", default="A cat holding a sign that says hello world"),
        guidance_scale: float = Input(description="Guidance scale", default=0.0),
        height: int = Input(description="Height of the generated image", default=768),
        width: int = Input(description="Width of the generated image", default=1360),
        num_inference_steps: int = Input(description="Number of inference steps", default=4),
        max_sequence_length: int = Input(description="Max sequence length", default=256),
    ) -> Path:
        out = self.pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
        ).images[0]
        
        output_path = Path("/tmp/image.png")
        out.save(output_path)
        return output_path
