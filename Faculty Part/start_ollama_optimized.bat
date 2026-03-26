@echo off
REM Ollama Performance Optimization Batch Script
REM Double-click this file to start Ollama with optimized settings

echo === Ollama Performance Optimization ===
echo.

REM Set environment variables for maximum performance
echo Setting Ollama environment variables...

REM Use all available GPUs
set OLLAMA_NUM_GPU=-1
echo [OK] OLLAMA_NUM_GPU = -1 (use all GPUs)

REM Allow multiple parallel requests
set OLLAMA_NUM_PARALLEL=4
echo [OK] OLLAMA_NUM_PARALLEL = 4 (handle 4 concurrent requests)

REM Keep only one model loaded at a time
set OLLAMA_MAX_LOADED_MODELS=1
echo [OK] OLLAMA_MAX_LOADED_MODELS = 1 (dedicate all VRAM to gemma3:12b)

REM Enable flash attention
set OLLAMA_FLASH_ATTENTION=1
echo [OK] OLLAMA_FLASH_ATTENTION = 1 (enable flash attention)

REM Set KV cache type
set OLLAMA_KV_CACHE_TYPE=q8_0
echo [OK] OLLAMA_KV_CACHE_TYPE = q8_0 (quantized cache)

echo.
echo === Starting Ollama ===
echo.
echo Press Ctrl+C to stop Ollama
echo.

REM Start Ollama serve
ollama serve
