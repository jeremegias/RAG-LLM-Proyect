import os
import shutil
import re
import random
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. GESTIÓN DE LLAVES (Rotación)
keys = [os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_API_KEY_2")]
api_keys_validas = [k for k in keys if k]

def obtener_llm():
    """Selecciona una API Key al azar y usa el modelo del commit funcional."""
    key_elegida = random.choice(api_keys_validas)
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", # EL MODELO QUE TE FUNCIONABA
        google_api_key=key_elegida,
        temperature=0.1
    )

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
DB_DIR = "./chroma_db"
HISTORIAL_FILE = "historial_resumido.txt"

# 2. CARGA DE DOCUMENTOS (Tu lógica estable)
loader = PyPDFDirectoryLoader("documentos/")
documentos_crudos = loader.load()

if len(documentos_crudos) > 0:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    fragmentos = text_splitter.split_documents(documentos_crudos)
    if os.path.exists(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        vectorstore.add_documents(fragmentos)
    else:
        vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory=DB_DIR)

    procesados_path = "documentos/procesados"
    if not os.path.exists(procesados_path): os.makedirs(procesados_path)
    for archivo in os.listdir("documentos/"):
        if archivo.endswith(".pdf"):
            shutil.move(os.path.join("documentos", archivo), os.path.join(procesados_path, archivo))
else:
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# --- 3. FUNCIONES DE LIMPIEZA Y PERSISTENCIA ---
def limpiar_respuesta(response):
    """Lógica de extracción de texto que tenías en el commit funcional."""
    if hasattr(response, 'content'):
        if isinstance(response.content, list):
            # Extrae el texto si viene como lista de bloques
            texto = "".join([b['text'] for b in response.content if 'text' in b])
        else:
            texto = response.content
    else:
        texto = str(response)
    
    # Limpieza extra de las firmas de Google (la 'mugre' técnica)
    texto = re.sub(r"'extras':\s*\{'signature':\s*'.*?'\}", "", texto)
    return texto.strip()

def cargar_memoria():
    if os.path.exists(HISTORIAL_FILE):
        with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def guardar_resumen(chat):
    if not chat: return
    try:
        llm_temp = obtener_llm()
        prompt_res = f"Resume en 3 líneas los puntos clave para recordarlos luego: {str(chat)}"
        res = llm_temp.invoke(prompt_res)
        with open(HISTORIAL_FILE, "w", encoding="utf-8") as f:
            f.write(f"Contexto previo: {limpiar_respuesta(res)}")
    except Exception as e:
        print(f"⚠️ Error al guardar historial: {e}")

# --- 4. INTERFAZ INICIAL ---
resumen_historico = cargar_memoria()
print("\n" + "="*50)
print("🔍 ANALISTA CONTABLE - MOTOR RESTAURADO (CON ROTACIÓN)")
print("="*50)

if resumen_historico:
    print(f"\n📚 MEMORIA DE SESIÓN:\n{resumen_historico}")
    print("-" * 50)
    opcion = input("👉 Elige tema, escribe 'nuevo' o 'salir': ").lower()
    if opcion in ['salir', 'exit', 'quit']: exit()
    if opcion == 'nuevo':
        if os.path.exists(HISTORIAL_FILE): os.remove(HISTORIAL_FILE)
        resumen_historico = ""
else:
    print("\n👋 No hay historial previo. Iniciando sesión nueva.")

ventana_chat = []

# --- 5. BUCLE DE CONSULTA ---
while True:
    try:
        pregunta = input("\n👤 Tu pregunta: ")
        
        # SALIDA Y GUARDADO
        if pregunta.lower() in ['salir', 'exit', 'quit']:
            print("📝 Guardando resumen y saliendo...")
            guardar_resumen(ventana_chat)
            break

        # Búsqueda (k=20 para no saturar pero ser preciso)
        docs = vectorstore.similarity_search(pregunta, k=20)
        contexto_pdfs = "\n\n".join([doc.page_content for doc in docs])
        chat_actual = "\n".join([f"{'Usuario' if isinstance(m, HumanMessage) else 'IA'}: {m.content}" for m in ventana_chat[-4:]])

        prompt = f"""
        ERES UN AUDITOR CONTABLE.
        RECUERDO ANTERIOR: {resumen_historico}
        CHAT RECIENTE: {chat_actual}
        DATOS PDFS: {contexto_pdfs}
        PREGUNTA: {pregunta}
        """

        llm_activo = obtener_llm()
        response = llm_activo.invoke(prompt)
        
        # Usamos la limpieza del commit funcional
        respuesta_final = limpiar_respuesta(response)

        print(f"\n🤖 IA:\n{respuesta_final}")
        ventana_chat.append(HumanMessage(content=pregunta))
        ventana_chat.append(AIMessage(content=respuesta_final))

    except Exception as e:
        if "429" in str(e):
            print("\n⚠️ API Saturada. Esperando 20 segundos...")
            time.sleep(20)
        else:
            print(f"\n❌ Error: {e}")