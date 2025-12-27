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

# 1. GESTIÓN DE LLAVES (Actualizado por consistencia)
keys_config = [
    {"name": "Key 1", "value": os.getenv("GOOGLE_API_KEY_1")},
    {"name": "Key 2", "value": os.getenv("GOOGLE_API_KEY_2")},
    {"name": "Key 3", "value": os.getenv("GOOGLE_API_KEY_3")}
]
# Filtramos solo las que realmente tengan un valor cargado en el .env
api_keys_validas = [k for k in keys_config if k["value"]]

def obtener_llm():
    seleccion = random.choice(api_keys_validas)
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        google_api_key=seleccion["value"],
        temperature=0.1
    )

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=api_keys_validas[0]["value"]
)
DB_DIR = "./chroma_db"
HISTORIAL_FILE = "historial_resumido.txt"

# 2. PROCESAMIENTO REFORZADO
if not os.path.exists("documentos"): os.makedirs("documentos")

loader = PyPDFDirectoryLoader("documentos/")
documentos_crudos = loader.load()

if os.path.exists(DB_DIR):
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    vectorstore = None

if len(documentos_crudos) > 0:
    print(f"📦 Detectados {len(documentos_crudos)} archivos. Procesando...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    fragmentos = text_splitter.split_documents(documentos_crudos)
    
    if vectorstore:
        vectorstore.add_documents(fragmentos)
    else:
        vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory=DB_DIR)

    procesados_path = "documentos/procesados"
    if not os.path.exists(procesados_path): os.makedirs(procesados_path)
    for archivo in os.listdir("documentos/"):
        if archivo.lower().endswith(".pdf"):
            shutil.move(os.path.join("documentos", archivo), os.path.join(procesados_path, archivo))
    print("✅ Archivos movidos a procesados.")
elif not vectorstore:
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# --- 3. FUNCIONES DE MEMORIA ---
def limpiar_respuesta(response):
    if hasattr(response, 'content'):
        texto = response.content if not isinstance(response.content, list) else "".join([b['text'] for b in response.content if 'text' in b])
    else:
        texto = str(response)
    return re.sub(r"'extras':\s*\{'signature':\s*'.*?'\}", "", texto).strip()

def cargar_memoria():
    if os.path.exists(HISTORIAL_FILE):
        with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def guardar_tema_prolijo(pregunta_usuario, ventana_chat):
    fecha = time.strftime('%d/%m %H:%M')
    tema_final = pregunta_usuario 
    
    try:
        llm_temp = obtener_llm()
        prompt_res = (
            f"Resume esta consulta en un título de máximo 5 palabras: '{pregunta_usuario}'"
        )
        res = llm_temp.invoke(prompt_res)
        tema_final = limpiar_respuesta(res).replace(".", "").strip()
    except Exception:
        tema_final = (pregunta_usuario[:40] + '...') if len(pregunta_usuario) > 40 else pregunta_usuario
        print("⚠️ Llaves saturadas: Usando título de respaldo.")

    try:
        with open(HISTORIAL_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n* [{fecha}] {tema_final}")
    except Exception as e:
        print(f"❌ Error al escribir historial: {e}")

# --- 4. INTERFAZ ---
resumen_historico = cargar_memoria()
print("\n" + "="*50)
print("🔍 ANALISTA CONTABLE - MONITOR ACTIVO")
print("="*50)

contexto_previo = ""
if resumen_historico:
    print(f"\n📚 TEMAS ANALIZADOS ANTERIORMENTE:")
    print(resumen_historico)
    print("-" * 50)
    opcion = input("👉 Elige un tema, escribe 'nuevo' (solo limpia historial) o 'salir': ").lower()
    
    if opcion in ['salir', 'exit', 'quit']: 
        exit()
    
    if opcion == 'nuevo':
        # SOLO borramos el archivo de texto del historial
        if os.path.exists(HISTORIAL_FILE): 
            os.remove(HISTORIAL_FILE)
        resumen_historico = ""
        print("✨ Historial de chat reiniciado. (Base de datos de PDFs conservada)")
    else:
        contexto_previo = f"Contexto previo: {opcion}"
else:
    print("\n👋 No hay registros previos. Usando base de datos existente.")
    contexto_previo = ""

# --- IMPORTANTE: DEFINIR LA LISTA DE CHAT AQUÍ ---
ventana_chat = [] 

# --- 5. BUCLE DE CONSULTA ---
while True:
    try:
        entrada_usuario = input("\n👤 Tu pregunta: ").strip()
        
        if entrada_usuario.lower() in ['salir', 'exit', 'quit']:
            print("👋 Saliendo...")
            import os as sistema_os
            sistema_os._exit(0)

        docs = vectorstore.similarity_search(entrada_usuario, k=15)
        contexto_pdfs = "\n\n".join([doc.page_content for doc in docs])
        chat_actual = "\n".join([f"{'U' if isinstance(m, HumanMessage) else 'IA'}: {m.content}" for m in ventana_chat[-4:]])
        
        instrucciones_estilo = (
            "Eres un extractor de datos contables. "
            "1. Presenta los datos en tabla: Fecha | Comercio | Importe. "
            "2. Suma el TOTAL al final. "
            "3. NO analices, solo extrae y suma."
            "4. Verifica que los consumos tengan distinto número de comprobante, para no repetirlos."
        )

        prompt = f"SISTEMA: {instrucciones_estilo}\n{contexto_previo}\nHISTORIAL RECIENTE: {chat_actual}\nPDFs: {contexto_pdfs}\nPREGUNTA: {entrada_usuario}"

        respuesta_recibida = False
        for llave_data in random.sample(api_keys_validas, len(api_keys_validas)):
            try:
                print(f"📡 Conectando vía: {llave_data['name']}")
                llm_intento = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=llave_data["value"], temperature=0.1)
                res = llm_intento.invoke(prompt)
                respuesta = limpiar_respuesta(res)
                
                print(f"\n🤖 IA:\n{respuesta}")
                ventana_chat.append(HumanMessage(content=entrada_usuario))
                ventana_chat.append(AIMessage(content=respuesta))
                
                guardar_tema_prolijo(entrada_usuario, ventana_chat)
                respuesta_recibida = True
                break 

            except Exception as e_api:
                if "429" in str(e_api):
                    print(f"⚠️ {llave_data['name']} saturada...")
                    time.sleep(2) # Pequeña pausa para no martillar la API
                    continue
                else: raise e_api

    except Exception as e:
        print(f"\n❌ Error: {e}")