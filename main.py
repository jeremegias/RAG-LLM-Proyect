import os
import requests
import time
import shutil  # <-- 1. Agregamos shutil para mover archivos
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. CONFIGURACIÓN DE LLAVE Y RUTAS
API_KEY = os.getenv("GOOGLE_API_KEY_1")
DB_DIR = "./chroma_db"
HISTORIAL_FILE = "historial_resumido.txt"
DOCS_PATH = "documentos"
PROCESADOS_PATH = os.path.join(DOCS_PATH, "procesados") # documentos/procesados

# Inicializar Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=API_KEY
)

# 2. PROCESAMIENTO DE DOCUMENTOS (Lógica de Respaldo Integrada)
if not os.path.exists(DOCS_PATH): os.makedirs(DOCS_PATH)
if not os.path.exists(PROCESADOS_PATH): os.makedirs(PROCESADOS_PATH)

loader = PyPDFDirectoryLoader(DOCS_PATH + "/")
documentos_crudos = loader.load()

# Inicializar vectorstore
if os.path.exists(DB_DIR):
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    vectorstore = None

if len(documentos_crudos) > 0:
    print(f"📦 Procesando {len(documentos_crudos)} archivos nuevos...")
    # Ajustamos para que los fragmentos sean más pequeños y cubran más hojas
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    fragmentos = text_splitter.split_documents(documentos_crudos)
    
    # 1. PRIMERO: Aseguramos la persistencia en la base de datos
    if vectorstore:
        vectorstore.add_documents(fragmentos)
    else:
        vectorstore = Chroma.from_documents(
            documents=fragmentos, 
            embedding=embeddings, 
            persist_directory=DB_DIR
        )
    
    # 2. SEGUNDO: Pausa de seguridad y movimiento de archivos
    time.sleep(2) 
    if not os.path.exists(PROCESADOS_PATH): os.makedirs(PROCESADOS_PATH)
    
    for archivo in os.listdir(DOCS_PATH):
        if archivo.lower().endswith(".pdf"):
            ruta_origen = os.path.join(DOCS_PATH, archivo)
            ruta_destino = os.path.join(PROCESADOS_PATH, archivo)
            try:
                shutil.move(ruta_origen, ruta_destino)
                print(f"✅ Documento procesado y movido: {archivo}")
            except Exception as e:
                print(f"⚠️ Error al mover {archivo}: {e}")
else:
    # 3. TERCERO: Si no hay archivos nuevos, cargamos lo que ya existe
    if not vectorstore:
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    print("ℹ️ No hay archivos nuevos. Usando base de datos existente.")

# 3. FUNCIONES CORE (Sin cambios, manteniendo k=15 y prompt limpio)
def consultar_auditor(mensaje, historial):
    try:
        docs = vectorstore.similarity_search(mensaje, k=15)
        contexto = "\n".join([d.page_content for d in docs])
    except:
        contexto = "No se pudo leer la base de datos."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={API_KEY}"
    
    prompt = f"""Actúa como auditor contable experto. Se inclusivo por defecto si hay por proximidad en el rubro y usa las notas para advertir. 
Verifica que los consumos unicamente tengan distinto número de comprobante evitar repetición. 
Usa este contexto de PDFs para responder:
{contexto}

Pregunta: {mensaje}"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        res_data = response.json()
        
        if "candidates" in res_data:
            respuesta = res_data['candidates'][0]['content']['parts'][0]['text']
            fecha = time.strftime('%d/%m %H:%M')
            with open(HISTORIAL_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n### {fecha}\n**Consulta:** {mensaje}\n")
            return respuesta
        else:
            error_msg = res_data.get('error', {}).get('message', 'Error de cuota')
            return f"⚠️ Google dice: {error_msg}. Espera 30 segundos y reintenta."
    except Exception as e:
        return f"❌ Error de conexión: {str(e)}"

def leer_historial():
    if os.path.exists(HISTORIAL_FILE):
        with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "No hay temas registrados."

# 4. INTERFAZ GRADIO
with gr.Blocks(theme=gr.themes.Soft(), title="Auditor Contable") as demo:
    gr.Markdown("# 🔍 Analista Contable IA\n### Auditoría con Gestión de Archivos")
    
    with gr.Tab("Chat de Auditoría"):
        gr.ChatInterface(fn=consultar_auditor)
    
    with gr.Tab("Historial de Temas"):
        output_h = gr.Markdown(value=leer_historial)
        btn_r = gr.Button("🔄 Actualizar Historial")
        btn_r.click(fn=leer_historial, outputs=output_h)

if __name__ == "__main__":
    demo.launch()