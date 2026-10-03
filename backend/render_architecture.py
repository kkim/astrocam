import os
import json
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/home/kio/projects/astrocam/backend/static"), name="static")

MD_FILE_PATH = "/home/kio/projects/astrocam/ARCHITECTURE.md"

@app.get("/content")
def get_content():
    if not os.path.exists(MD_FILE_PATH):
        return {"content": "ARCHITECTURE.md not found.", "last_modified": 0}
    
    mtime = os.path.getmtime(MD_FILE_PATH)
    with open(MD_FILE_PATH, "r") as f:
        content = f.read()
    return {"content": content, "last_modified": mtime}

@app.get("/", response_class=HTMLResponse)
def serve_page():
    html_content = r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AstroCam Architecture Viewer</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <!-- KaTeX for equation rendering -->
        <link rel="stylesheet" href="/static/katex.min.css">
        <script src="/static/katex.min.js"></script>
        <script src="/static/contrib/auto-render.min.js"></script>
        <!-- Standalone Mermaid JS -->
        <script src="/static/mermaid.min.js"></script>
        <style>
            :root {
                --bg-dark: #0d1117;
                --bg-sidebar: #161b22;
                --bg-panel: #21262d;
                --border-color: #30363d;
                --text-primary: #c9d1d9;
                --text-muted: #8b949e;
                --accent: #58a6ff;
                --accent-emerald: #10b981;
            }

            body {
                margin: 0;
                padding: 0;
                background-color: var(--bg-dark);
                color: var(--text-primary);
                font-family: 'Outfit', sans-serif;
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }

            header {
                background-color: var(--bg-sidebar);
                border-bottom: 1px solid var(--border-color);
                padding: 16px 24px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }

            h1 {
                margin: 0;
                font-size: 1.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, var(--accent), var(--accent-emerald));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .badge {
                background-color: var(--bg-panel);
                border: 1px solid var(--border-color);
                padding: 4px 10px;
                border-radius: 9999px;
                font-size: 0.8rem;
                color: var(--text-muted);
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .badge .dot {
                width: 8px;
                height: 8px;
                background-color: var(--accent-emerald);
                border-radius: 50%;
                display: inline-block;
                animation: pulse 1.5s infinite;
            }

            @keyframes pulse {
                0% { opacity: 0.3; }
                50% { opacity: 1; }
                100% { opacity: 0.3; }
            }

            main {
                flex: 1;
                display: flex;
                padding: 24px;
                gap: 24px;
                box-sizing: border-box;
                height: calc(100vh - 70px);
            }

            section {
                flex: 1;
                background-color: var(--bg-sidebar);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 24px;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            }

            .panel-header {
                font-size: 1.1rem;
                font-weight: 600;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 12px;
                margin-bottom: 16px;
                color: var(--accent);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .diagram-container {
                flex: 1;
                overflow: auto;
                background-color: var(--bg-dark);
                border-radius: 8px;
                border: 1px solid var(--border-color);
                padding: 16px;
            }

            .diagram-container svg {
                width: 100% !important;
                height: auto !important;
                max-width: 100% !important;
                display: block;
                margin: 0 auto;
            }

            .markdown-container {
                flex: 1;
                overflow-y: auto;
                font-size: 0.95rem;
                line-height: 1.6;
                padding-right: 8px;
            }

            .markdown-container::-webkit-scrollbar {
                width: 6px;
            }

            .markdown-container::-webkit-scrollbar-thumb {
                background-color: var(--border-color);
                border-radius: 3px;
            }

            pre {
                background-color: var(--bg-dark);
                padding: 12px;
                border-radius: 6px;
                border: 1px solid var(--border-color);
                overflow-x: auto;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
            }

            h2, h3 {
                color: white;
                margin-top: 24px;
            }

            code {
                font-family: 'JetBrains Mono', monospace;
                background-color: var(--bg-dark);
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.85rem;
                border: 1px solid var(--border-color);
            }
        </style>
    </head>
    <body>
        <header>
            <h1>🔭 AstroCam Architecture</h1>
            <div class="badge">
                <span class="dot"></span> Live Monitoring (2s Interval)
            </div>
        </header>
        <main>
            <section style="flex: 1.3;">
                <div class="panel-header">Mermaid Diagram</div>
                <div class="diagram-container" id="diagram-view">
                    <div style="color: var(--text-muted);">Loading diagram...</div>
                </div>
            </section>
            <section style="flex: 1;">
                <div class="panel-header">Documentation Overview</div>
                <div class="markdown-container" id="doc-view">
                    <div style="color: var(--text-muted);">Loading documentation...</div>
                </div>
            </section>
        </main>

        <script>
            // Initialize mermaid with custom dark theme configurations
            mermaid.initialize({
                startOnLoad: false,
                theme: 'dark',
                securityLevel: 'loose',
                themeVariables: {
                    background: '#161b22',
                    primaryColor: '#21262d',
                    primaryTextColor: '#c9d1d9',
                    lineColor: '#30363d',
                    secondaryColor: '#0d1117'
                }
            });

            let lastModified = 0;

            // Simple markdown parser for description
            function parseMarkdown(text) {
                // Remove the mermaid diagram block for doc view
                let docText = text.replace(/```mermaid[\s\S]*?```/, '<div style="background:var(--bg-dark); padding:12px; border-radius:6px; border:1px solid var(--border-color); color:var(--text-muted); font-size:0.9rem; text-align:center; margin-bottom:16px;">[Mermaid Diagram Rendered in Left Panel]</div>');
                
                // Headers
                docText = docText.replace(/^# (.*$)/gim, '<h1>$1</h1>');
                docText = docText.replace(/^## (.*$)/gim, '<h2>$1</h2>');
                docText = docText.replace(/^### (.*$)/gim, '<h3>$1</h3>');
                
                // Bold
                docText = docText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                
                // Code block
                docText = docText.replace(/`(.*?)`/g, '<code>$1</code>');
                
                // Unordered list
                docText = docText.replace(/^\* (.*$)/gim, '<li>$1</li>');
                
                // Line breaks
                docText = docText.replace(/\n/g, '<br>');

                return docText;
            }

            async function checkAndUpdate() {
                try {
                    const res = await fetch('/content');
                    if (!res.ok) return;
                    const data = await res.json();
                    
                    if (data.last_modified !== lastModified) {
                        lastModified = data.last_modified;
                        
                        // Render documentation
                        document.getElementById('doc-view').innerHTML = parseMarkdown(data.content);

                        // Render equations using KaTeX
                        if (window.renderMathInElement) {
                            window.renderMathInElement(document.getElementById('doc-view'), {
                                delimiters: [
                                    {left: '$$', right: '$$', display: true},
                                    {left: '$', right: '$', display: false}
                                ]
                            });
                        }
                        
                        // Extract mermaid block
                        const mermaidMatch = data.content.match(/```mermaid([\s\S]*?)```/);
                        if (mermaidMatch && mermaidMatch[1]) {
                            const graphDefinition = mermaidMatch[1].trim();
                            const container = document.getElementById('diagram-view');
                            
                            // Re-render mermaid
                            container.innerHTML = '<div class="mermaid" id="graph">' + graphDefinition + '</div>';
                            try {
                                await mermaid.run({
                                    nodes: [document.getElementById('graph')]
                                });
                            } catch (err) {
                                container.innerHTML = '<div style="color: var(--danger); font-family: monospace;">Render Error: ' + err.message + '</div>';
                                console.error(err);
                            }
                        }
                    }
                } catch (err) {
                    console.error("Poll error:", err);
                }
            }

            // Initial load and start poll
            setInterval(checkAndUpdate, 2000);
            checkAndUpdate();
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
