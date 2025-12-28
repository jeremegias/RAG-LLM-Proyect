import os
import requests
import time
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

# Inicializar Embeddings para los PDFs
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=API_KEY
)

# 2. PROCESAMIENTO DE DOCUMENTOS
if not os.path.exists("documentos"): os.makedirs("documentos")
loader = PyPDFDirectoryLoader("documentos/")
documentos_crudos = loader.load()

if documentos_crudos:
    print("📦 Procesando PDFs nuevos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    fragmentos = text_splitter.split_documents(documentos_crudos)
    vectorstore = Chroma.from_documents(
        documents=fragmentos, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print("✅ PDFs integrados a la base de datos.")
else:
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# 3. FUNCIONES CORE
def consultar_auditor(mensaje, historial):
    # Recuperar información de los PDFs
    try:
        docs = vectorstore.similarity_search(mensaje, k=15)
        contexto = "\n".join([d.page_content for d in docs])
    except:
        contexto = "No se pudo leer la base de datos."
    
    # CAMBIAMOS A GEMINI 3 FLASH (El que te funcionó en la web)
    # Nota: Usamos v1beta porque los modelos 'Preview' o '3' lo requieren
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={API_KEY}"
    
    prompt = f"""Actúa como auditor contable experto. 
    Verifica que los consumos tengan distinto número de comprobante para no repetirlos. 
    Usa este contexto de PDFs para responder:\n{contexto}\n\nPregunta: {mensaje}"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        res_data = response.json()
        
        if "candidates" in res_data:
            respuesta = res_data['candidates'][0]['content']['parts'][0]['text']
            # Guardar en historial
            fecha = time.strftime('%d/%m %H:%M')
            with open(HISTORIAL_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n### {fecha}\n**Consulta:** {mensaje}\n")
            return respuesta
        else:
            # Si Gemini 3 también falla, probamos el 1.5 Flash como último recurso
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
    gr.Markdown("# 🔍 Analista Contable IA\n### Auditoría con PDFs e Historial")
    
    with gr.Tab("Chat de Auditoría"):
        gr.ChatInterface(fn=consultar_auditor)
    
    with gr.Tab("Historial de Temas"):
        output_h = gr.Markdown(value=leer_historial)
        btn_r = gr.Button("🔄 Actualizar Historial")
        btn_r.click(fn=leer_historial, outputs=output_h)

if __name__ == "__main__":
    demo.launch()