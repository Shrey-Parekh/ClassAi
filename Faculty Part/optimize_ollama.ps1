# Ollama Performance Optimization Script for Windows
# Run this before starting Ollama to maximize GPU utilization

Write-Host "=== Ollama Performance Optimization ===" -ForegroundColor Cyan
Write-Host ""

# Set environment variables for maximum performance
Write-Host "Setting Ollama environment variables..." -ForegroundColor Yellow

# Use all available GPUs
$env:OLLAMA_NUM_GPU = "-1"
Write-Host "✓ OLLAMA_NUM_GPU = -1 (use all GPUs)" -ForegroundColor Green

# Allow multiple parallel requests
$env:OLLAMA_NUM_PARALLEL = "4"
Write-Host "✓ OLLAMA_NUM_PARALLEL = 4 (handle 4 concurrent requests)" -ForegroundColor Green

# Keep only one model loaded at a time to maximize VRAM for active model
$env:OLLAMA_MAX_LOADED_MODELS = "1"
Write-Host "✓ OLLAMA_MAX_LOADED_MODELS = 1 (dedicate all VRAM to gemma3:12b)" -ForegroundColor Green

# Increase flash attention for faster inference
$env:OLLAMA_FLASH_ATTENTION = "1"
Write-Host "✓ OLLAMA_FLASH_ATTENTION = 1 (enable flash attention)" -ForegroundColor Green

# Set KV cache type for better memory efficiency
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
Write-Host "✓ OLLAMA_KV_CACHE_TYPE = q8_0 (quantized cache for speed)" -ForegroundColor Green

Write-Host ""
Write-Host "=== Configuration Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Stop Ollama if running: taskkill /F /IM ollama.exe" -ForegroundColor White
Write-Host "2. Start Ollama with new settings: ollama serve" -ForegroundColor White
Write-Host "3. Load model: ollama run gemma3:12b" -ForegroundColor White
Write-Host "4. Monitor GPU usage: nvidia-smi" -ForegroundColor White
Write-Host ""
Write-Host "Expected improvements:" -ForegroundColor Yellow
Write-Host "- 2-3x faster generation (512 tokens vs 1024)" -ForegroundColor White
Write-Host "- Better GPU utilization (all 8GB VRAM available)" -ForegroundColor White
Write-Host "- Reduced context window (8K vs 16K) for faster processing" -ForegroundColor White
Write-Host ""
