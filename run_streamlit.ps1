$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

& "C:\Users\sanch\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
    -m streamlit run streamlit_app.py `
    --server.headless true `
    --browser.gatherUsageStats false `
    --server.address 127.0.0.1 `
    --server.port 8501
