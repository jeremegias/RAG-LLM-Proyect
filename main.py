import os
import shutil
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 1. CONFIGURACIÓN (Mantenemos gemini-3-flash-preview)
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 2. CARGA DE DOCUMENTOS
loader = PyPDFDirectoryLoader("documentos/")
documentos_crudos = loader.load()
DB_DIR = "./chroma_db"

if len(documentos_crudos) > 0:
    print(f"📄 Se han encontrado {len(documentos_crudos)} páginas nuevas.")
    # Ajuste de precisión que ya te funcionó
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    fragmentos = text_splitter.split_documents(documentos_crudos)
    
    if os.path.exists(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        vectorstore.add_documents(fragmentos)
    else:
        vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory=DB_DIR)

    # Lógica de archivado
    procesados_path = "documentos/procesados"
    if not os.path.exists(procesados_path): os.makedirs(procesados_path)
    for archivo in os.listdir("documentos/"):
        if archivo.endswith(".pdf"):
            shutil.move(os.path.join("documentos", archivo), os.path.join(procesados_path, archivo))
    print("📂 PDFs archivados en 'procesados'.")
else:
    if os.path.exists(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        print("❌ Error: Sin documentos ni base de datos."); exit()

# --- 5. BUCLE DE CHAT CON HISTORIAL ---
print("\n--- 🤖 Analista con Memoria de Chat listo ---")
historial = []  # Lista para guardar el contexto de la conversación

while True:
    try:
        pregunta = input("\n👤 Tu pregunta: ")
        if pregunta.lower() in ['salir', 'exit', 'quit']: break

        # Búsqueda en los PDFs (k=15 para no perder datos)
        docs_relevantes = vectorstore.similarity_search(pregunta, k=15)
        contexto_pdfs = "\n\n".join([doc.page_content for doc in docs_relevantes])

        # Construimos el bloque de historial para el prompt (últimos 6 mensajes)
        texto_historial = ""
        for msg in historial[-6:]:
            autor = "Usuario" if isinstance(msg, HumanMessage) else "IA"
            texto_historial += f"{autor}: {msg.content}\n"

        prompt = f"""
        ERES UN AUDITOR CONTABLE QUE RECUERDA LA CONVERSACIÓN.
        
        HISTORIAL RECIENTE:
        {texto_historial}
        
        CONTEXTO DE LOS PDFS:
        {contexto_pdfs}

        NUEVA PREGUNTA: {pregunta}
        
        REGLAS:
        1. Usa el historial para entender referencias como "ese gasto", "el mes anterior" o "¿quién es el titular?".
        2. Si la información no está en el historial, búscala en el contexto de los PDFs.
        3. Mantén el detalle de auditoría (fechas y montos individuales).
        """
        
        response = llm.invoke(prompt)
        
        # Limpieza de respuesta (evita corchetes/firmas)
        if hasattr(response, 'content'):
            if isinstance(response.content, list):
                respuesta_final = "".join([b['text'] for b in response.content if 'text' in b])
            else:
                respuesta_final = response.content
        else:
            respuesta_final = str(response)

        print(f"\n🤖 IA:\n{respuesta_final}")

        # Guardamos la interacción en el historial
        historial.append(HumanMessage(content=pregunta))
        historial.append(AIMessage(content=respuesta_final))

    except Exception as e:
        print(f"\n❌ Error: {e}")