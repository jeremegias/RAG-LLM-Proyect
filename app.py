import os
import requests
import time
import shutil
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. CONFIGURACIÓN DE LLAVE Y RUTAS
API_KEY = os.getenv("GOOGLE_API_KEY_2")
DB_DIR = "./chroma_db"
HISTORIAL_FILE = "historial_resumido.txt"
DOCS_PATH = "documentos"
PROCESADOS_PATH = os.path.join(DOCS_PATH, "procesados")

# --- NUEVO: ESTILO CSS PARA IPHONE ---
custom_css = """
.gradio-container table {
    display: table !important;
    width: 100% !important;
    overflow-x: auto !important;
    border-collapse: collapse !important;
}
.gradio-container th, .gradio-container td {
    white-space: nowrap !important;
    padding: 8px !important;
    border: 1px solid #ddd !important;
    font-size: 13px !important;
}
"""

# Inicializar Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=API_KEY
)

# 2. PROCESAMIENTO DE DOCUMENTOS
if not os.path.exists(DOCS_PATH): os.makedirs(DOCS_PATH)
if not os.path.exists(PROCESADOS_PATH): os.makedirs(PROCESADOS_PATH)

loader = PyPDFDirectoryLoader(DOCS_PATH + "/")
documentos_crudos = loader.load()

if os.path.exists(DB_DIR):
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    vectorstore = None

if len(documentos_crudos) > 0:
    print(f"📦 Procesando {len(documentos_crudos)} archivos nuevos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    fragmentos = text_splitter.split_documents(documentos_crudos)
    
    if vectorstore:
        vectorstore.add_documents(fragmentos)
    else:
        vectorstore = Chroma.from_documents(
            documents=fragmentos, 
            embedding=embeddings, 
            persist_directory=DB_DIR
        )
    
    time.sleep(2) 
    for archivo in os.listdir(DOCS_PATH):
        if archivo.lower().endswith(".pdf"):
            ruta_origen = os.path.join(DOCS_PATH, archivo)
            ruta_destino = os.path.join(PROCESADOS_PATH, archivo)
            try:
                shutil.move(ruta_origen, ruta_destino)
            except Exception as e:
                print(f"⚠️ Error al mover {archivo}: {e}")
else:
    if not vectorstore:
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# 3. FUNCIONES CORE (MODIFICADA CON SEGURIDAD)
# Agregamos 'request: gr.Request' para leer el token de la URL
def consultar_auditor(mensaje, historial, request: gr.Request):
    # Verificación de Token
    token_valido = "EJnmxd89"
    token_usuario = request.query_params.get("token")
    
    if token_usuario != token_valido:
        return "⚠️ ACCESO DENEGADO. Por favor, ingresa mediante el link autorizado con su token de seguridad."

    try:
        docs = vectorstore.similarity_search(mensaje, k=30)
        contexto_lista = []
        for d in docs:
            fuente = os.path.basename(d.metadata.get('source', 'Archivo desconocido'))
            bloque = f"DOCUMENTO: {fuente}\nCONTENIDO: {d.page_content}"
            contexto_lista.append(bloque)
        contexto = "\n\n---\n\n".join(contexto_lista)
    except:
        contexto = "No se pudo leer la base de datos."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={API_KEY}"
    
    historial_google = []
    for m in historial:
        rol = "user" if m["role"] == "user" else "model"
        texto_limpio = str(m["content"]) 
        historial_google.append({"role": rol, "parts": [{"text": texto_limpio}]})
    
    prompt_final = f"""Actúa como auditor contable experto. Se inclusivo por defecto si hay proximidad en el rubro y usa las notas para advertir. 
Verifica que los consumos tengan distinto número de comprobante para evitar repetición. 
Usa este contexto de PDFs para responder:
{contexto}

Pregunta: {mensaje}"""

    payload = {
        "contents": historial_google + [{"role": "user", "parts": [{"text": prompt_final}]}]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        res_data = response.json()
        if "candidates" in res_data:
            respuesta = res_data['candidates'][0]['content']['parts'][0]['text']
            with open(HISTORIAL_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n### {time.strftime('%d/%m %H:%M')}\n**Consulta:** {mensaje}\n")
            return respuesta
        else:
            return "⚠️ Error de respuesta de Google."
    except Exception as e:
        return f"❌ Error de conexión: {str(e)}"

def leer_historial():
    if os.path.exists(HISTORIAL_FILE):
        with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "No hay temas registrados."

# 4. INTERFAZ GRADIO
with gr.Blocks(title="Auditor Contable") as demo:
    gr.Markdown("# 🔍 Analista Contable IA\n### Acceso Restringido")
    
    with gr.Tab("Chat de Auditoría"):
        gr.ChatInterface(fn=consultar_auditor)
    
    with gr.Tab("Historial de Temas"):
        output_h = gr.Markdown(value=leer_historial)
        btn_r = gr.Button("🔄 Actualizar Historial")
        btn_r.click(fn=leer_historial, outputs=output_h)

if __name__ == "__main__":
    demo.launch(
        css=custom_css
    )