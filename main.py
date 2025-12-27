import os
import shutil
import re
import random
import time
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. GESTIÓN DE LLAVES
keys_config = [
    {"name": "Key 1", "value": os.getenv("GOOGLE_API_KEY_1")},
    {"name": "Key 2", "value": os.getenv("GOOGLE_API_KEY_2")},
    {"name": "Key 3", "value": os.getenv("GOOGLE_API_KEY_3")}
]
api_keys_validas = [k for k in keys_config if k["value"]]

def obtener_llm():
    seleccion = random.choice(api_keys_validas)
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        google_api_key=seleccion["value"],
        temperature=0.1,
        convert_system_message_to_human=True
    )

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=api_keys_validas[0]["value"]
)
DB_DIR = "./chroma_db"
HISTORIAL_FILE = "historial_resumido.txt"

# 2. PROCESAMIENTO DE PDFS
if not os.path.exists("documentos"): os.makedirs("documentos")
loader = PyPDFDirectoryLoader("documentos/")
documentos_crudos = loader.load()

if os.path.exists(DB_DIR):
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    vectorstore = None

if len(documentos_crudos) > 0:
    print(f"📦 Procesando nuevos archivos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    fragmentos = text_splitter.split_documents(documentos_crudos)
    if vectorstore: vectorstore.add_documents(fragmentos)
    else: vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory=DB_DIR)
    
    procesados_path = "documentos/procesados"
    if not os.path.exists(procesados_path): os.makedirs(procesados_path)
    for archivo in os.listdir("documentos/"):
        if archivo.lower().endswith(".pdf"):
            shutil.move(os.path.join("documentos", archivo), os.path.join(procesados_path, archivo))
elif not vectorstore:
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# 3. FUNCIONES DE APOYO
def limpiar_respuesta(response):
    if hasattr(response, 'content'):
        texto = response.content if not isinstance(response.content, list) else "".join([b['text'] for b in response.content if 'text' in b])
    else: texto = str(response)
    return re.sub(r"'extras':\s*\{'signature':\s*'.*?'\}", "", texto).strip()

def guardar_tema_prolijo(pregunta_usuario):
    fecha = time.strftime('%d/%m %H:%M')
    # En lugar de gastar una llamada a la IA para resumir, 
    # usemos los primeros 30 caracteres directamente:
    tema = pregunta_usuario[:30] + "..."
     
    with open(HISTORIAL_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n* [{fecha}] {tema}")

# --- 4. FUNCIÓN CORE PARA GRADIO ---
def responder_auditoria(mensaje, historial):
    # 1. Recuperamos contexto
    docs = vectorstore.similarity_search(mensaje, k=15)
    contexto_pdfs = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Instrucciones
    instrucciones = (
        "Eres un extractor de datos contables de precisión. "
        "1. Presenta los datos en tabla: Fecha | Comercio | Importe. "
        "2. Suma el TOTAL al final. "
        "3. NO analices, solo extrae y suma. "
        "4. Verifica que los consumos tengan distinto número de comprobante para no repetirlos."
    )
    
    prompt = f"SISTEMA: {instrucciones}\nPDFs: {contexto_pdfs}\nPREGUNTA: {mensaje}"
    
    # 3. Bucle de llaves con el modelo 2.0
    for llave_data in random.sample(api_keys_validas, len(api_keys_validas)):
        try:
            print(f"📡 Intentando con: {llave_data['name']} (Gemini 2.0)")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp", 
                google_api_key=llave_data["value"],
                convert_system_message_to_human=True
            )
            res = llm.invoke(prompt)
            respuesta = limpiar_respuesta(res)
            guardar_tema_prolijo(mensaje)
            return respuesta
            
        except Exception as e:
            if "429" in str(e):
                continue
            else:
                return f"Error técnico: {str(e)}"
                
    return "⚠️ Todas las llaves saturadas. Reintenta en breve."

# 5. INTERFAZ GRADIO
with gr.Blocks(theme=gr.themes.Soft(), title="Auditor Contable") as demo:
    gr.Markdown("# 🔍 Analista Contable IA\n### Auditoría con Gemini 2.0 Flash")
    
    with gr.Tab("Chat de Auditoría"):
        gr.ChatInterface(
            fn=responder_auditoria,
            examples=["Suma el combustible de diciembre", "Tabla de gastos de servicios"],
            cache_examples=False
        )
    
    with gr.Tab("Historial de Temas"):
        output_historial = gr.Markdown(value="Haz clic en actualizar para ver temas anteriores.")
        btn_refresh = gr.Button("Actualizar Historial")
        
        def ver_historial():
            if os.path.exists(HISTORIAL_FILE):
                with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
                    return f.read()
            return "No hay historial aún."
        
        btn_refresh.click(fn=ver_historial, outputs=output_historial)

if __name__ == "__main__":
    demo.launch(share=True)