import os
import shutil
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. CONFIGURACIÓN
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
DB_DIR = "./chroma_db"
HISTORIAL_FILE = "historial_resumido.txt"

# 2. CARGA DE DOCUMENTOS (Mantenemos tu lógica estable)
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
    
    # Archivados
    procesados_path = "documentos/procesados"
    if not os.path.exists(procesados_path): os.makedirs(procesados_path)
    for archivo in os.listdir("documentos/"):
        if archivo.endswith(".pdf"):
            shutil.move(os.path.join("documentos", archivo), os.path.join(procesados_path, archivo))
else:
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# --- 3. LÓGICA DE MEMORIA RESUMIDA PERSISTENTE ---
def cargar_memoria_previa():
    if os.path.exists(HISTORIAL_FILE):
        with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "No hay conversaciones previas."

def guardar_resumen(nuevo_resumen):
    with open(HISTORIAL_FILE, "w", encoding="utf-8") as f:
        f.write(nuevo_resumen)

# Cargamos lo que recordamos de sesiones anteriores
resumen_historico = cargar_memoria_previa()
ventana_chat = [] # Chat activo de la sesión

print("\n--- 🤖 Analista con Memoria Persistente ---")
print(f"🧠 Recuerdo anterior: {resumen_historico[:100]}...")

while True:
    try:
        pregunta = input("\n👤 Tu pregunta: ")
        if pregunta.lower() in ['salir', 'exit', 'quit']:
            # ANTES DE SALIR: Resumimos la sesión para la próxima vez
            print("📝 Resumiendo conversación para el próximo inicio...")
            prompt_resumen = f"Resume en 3 líneas los puntos clave de esta charla para recordarlos luego: {str(ventana_chat)}"
            resumen_final = llm.invoke(prompt_resumen).content
            guardar_resumen(f"Contexto previo: {resumen_final}")
            break

        docs_relevantes = vectorstore.similarity_search(pregunta, k=15)
        contexto_pdfs = "\n\n".join([doc.page_content for doc in docs_relevantes])

        # Construimos historial de la sesión actual
        chat_actual = "\n".join([f"{'Usuario' if isinstance(m, HumanMessage) else 'IA'}: {m.content}" for m in ventana_chat[-4:]])

        prompt = f"""
        ERES UN AUDITOR CONTABLE.
        
        LO QUE RECORDAMOS DE DÍAS ANTERIORES:
        {resumen_historico}
        
        LO QUE HABLAMOS RECIÉN:
        {chat_actual}
        
        DATOS DE LOS PDFS:
        {contexto_pdfs}

        PREGUNTA: {pregunta}
        """
        
        response = llm.invoke(prompt)
        respuesta_final = response.content if not isinstance(response.content, list) else "".join([b['text'] for b in response.content if 'text' in b])

        print(f"\n🤖 IA:\n{respuesta_final}")

        ventana_chat.append(HumanMessage(content=pregunta))
        ventana_chat.append(AIMessage(content=respuesta_final))

    except Exception as e:
        print(f"\n❌ Error: {e}")