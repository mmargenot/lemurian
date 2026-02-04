"""
Modal deployment script for vLLM.

This script deploys vLLM as a serverless endpoint on Modal, providing an
OpenAI-compatible API that can be used with ModalVLLMProvider.

Usage:
    # Deploy the vLLM server to Modal
    modal deploy src/scripts/modal_vllm.py

    # For development/testing (runs temporarily)
    modal serve src/scripts/modal_vllm.py

    # Run with a different model
    MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" modal deploy src/scripts/modal_vllm.py

Environment Variables:
    MODEL_ID: HuggingFace model ID to serve (default: Qwen/Qwen3-8B)
    GPU_TYPE: Modal GPU type (default: A100)
    GPU_COUNT: Number of GPUs for tensor parallelism (default: 1)
    MAX_MODEL_LEN: Maximum sequence length (default: 8192)
    HF_TOKEN: HuggingFace token for gated models (optional)
"""

import os
import modal

# Configuration from environment variables
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
GPU_TYPE = os.environ.get("GPU_TYPE", "A100")
GPU_COUNT = int(os.environ.get("GPU_COUNT", "1"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))

# Tool call parser mapping for common models
TOOL_CALL_PARSERS = {
    "qwen": "hermes",
    "llama": "llama3_json",
    "mistral": "mistral",
    "hermes": "hermes",
}

REASONING_PARSERS = {
    "qwen3": "qwen3",
    "deepseek": "deepseek_r1",
}


def get_tool_call_parser(model_id: str) -> str:
    """Determine the appropriate tool call parser based on model name."""
    model_lower = model_id.lower()
    for key, parser in TOOL_CALL_PARSERS.items():
        if key in model_lower:
            return parser
    return "hermes"  # Default fallback


def get_reasoning_parser(model_id: str) -> str | None:
    """Determine if a reasoning parser is needed based on model name."""
    model_lower = model_id.lower()
    for key, parser in REASONING_PARSERS.items():
        if key in model_lower:
            return parser
    return None


# Create Modal app
app = modal.App("lemurian-vllm")

# Create a volume for caching model weights
model_cache = modal.Volume.from_name("lemurian-model-cache", create_if_missing=True)

# Build the container image with vLLM
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm>=0.8.0",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.cls(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={"/root/.cache/huggingface": model_cache},
    timeout=60 * 60,  # 1 hour timeout
    container_idle_timeout=300,  # 5 minutes idle before scaling down
    allow_concurrent_inputs=100,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
class VLLMServer:
    model_id: str = MODEL_ID

    @modal.enter()
    def start_engine(self):
        """Initialize vLLM engine when container starts."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        # Determine parsers based on model
        tool_parser = get_tool_call_parser(self.model_id)
        reasoning_parser = get_reasoning_parser(self.model_id)

        engine_args_dict = {
            "model": self.model_id,
            "max_model_len": MAX_MODEL_LEN,
            "tensor_parallel_size": GPU_COUNT,
            "enable_auto_tool_choice": True,
            "tool_call_parser": tool_parser,
            "trust_remote_code": True,
        }

        if reasoning_parser:
            engine_args_dict["reasoning_parser"] = reasoning_parser

        engine_args = AsyncEngineArgs(**engine_args_dict)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Store config for the OpenAI server
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser

    @modal.asgi_app()
    def serve(self):
        """Serve vLLM with OpenAI-compatible API."""
        from vllm.entrypoints.openai.api_server import (
            build_async_engine_client,
            init_app_state,
        )
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        import fastapi

        # Create FastAPI app
        app = fastapi.FastAPI(
            title="Lemurian vLLM Server",
            description="OpenAI-compatible API for vLLM on Modal",
        )

        # Build argument parser and create args
        parser = make_arg_parser()
        args_list = [
            "--model", self.model_id,
            "--max-model-len", str(MAX_MODEL_LEN),
            "--tensor-parallel-size", str(GPU_COUNT),
            "--enable-auto-tool-choice",
            "--tool-call-parser", self.tool_parser,
            "--trust-remote-code",
        ]

        if self.reasoning_parser:
            args_list.extend(["--reasoning-parser", self.reasoning_parser])

        args = parser.parse_args(args_list)

        # Import and include the OpenAI API router
        from vllm.entrypoints.openai.api_server import (
            router as openai_router,
            mount_metrics,
        )

        app.include_router(openai_router)

        @app.on_event("startup")
        async def startup():
            async with build_async_engine_client(args) as engine_client:
                await init_app_state(
                    engine_client,
                    args,
                    app.state,
                )

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": self.model_id}

        @app.get("/")
        async def root():
            return {
                "message": "Lemurian vLLM Server",
                "model": self.model_id,
                "docs": "/docs",
            }

        return app


@app.function(image=vllm_image, volumes={"/root/.cache/huggingface": model_cache})
def download_model(model_id: str = MODEL_ID):
    """Pre-download model weights to the cache volume."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        model_id,
        ignore_patterns=["*.pt", "*.bin"],  # Prefer safetensors
    )
    model_cache.commit()
    print(f"Successfully downloaded {model_id}")


@app.local_entrypoint()
def main(download: bool = False):
    """
    Local entrypoint for the Modal app.

    Args:
        download: If True, download model weights before deployment
    """
    if download:
        print(f"Downloading model: {MODEL_ID}")
        download_model.remote(MODEL_ID)
        print("Download complete!")
    else:
        print(f"Deploy with: modal deploy {__file__}")
        print(f"Or serve temporarily with: modal serve {__file__}")
        print(f"\nCurrent configuration:")
        print(f"  Model: {MODEL_ID}")
        print(f"  GPU: {GPU_TYPE} x {GPU_COUNT}")
        print(f"  Max sequence length: {MAX_MODEL_LEN}")
