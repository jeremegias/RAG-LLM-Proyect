import os
import shutil
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

# 1. CONFIGURACIÓN DEL MODELO (Versión estable recuperada)
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 2. CARGA DE DOCUMENTOS
if not os.path.exists("documentos"):
    os.makedirs("documentos")

loader = PyPDFDirectoryLoader("documentos/")
documentos_crudos = loader.load()

# Directorio de persistencia
DB_DIR = "./chroma_db"

if len(documentos_crudos) > 0:
    print(f"📄 Se han encontrado {len(documentos_crudos)} páginas nuevas.")
    
    # 3. FRAGMENTACIÓN (Ajustado a 2000 para no cortar tablas de gastos)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    fragmentos = text_splitter.split_documents(documentos_crudos)
    
    # 4. ACTUALIZACIÓN O CREACIÓN DE BASE DE DATOS
    if os.path.exists(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        vectorstore.add_documents(fragmentos)
        print("✅ Base de datos actualizada con la nueva información.")
    else:
        vectorstore = Chroma.from_documents(
            documents=fragmentos, 
            embedding=embeddings, 
            persist_directory=DB_DIR
        )
        print("🆕 Base de datos creada desde cero.")

    # --- LÓGICA DE MOVIMIENTO A PROCESADOS ---
    procesados_path = "documentos/procesados"
    if not os.path.exists(procesados_path):
        os.makedirs(procesados_path)
    
    for archivo in os.listdir("documentos/"):
        if archivo.endswith(".pdf"):
            shutil.move(os.path.join("documentos", archivo), os.path.join(procesados_path, archivo))
    print("📂 PDFs movidos a 'procesados' para evitar duplicados en el próximo inicio.")

else:
    if os.path.exists(DB_DIR):
        print("ℹ️ No hay archivos nuevos. Usando la memoria existente.")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        print("❌ Error: No hay documentos en 'documentos/' ni base de datos previa."); exit()

# 5. BUCLE DE CHAT
print("\n--- 🤖 Analista Contable Listo ---")
while True:
    try:
        pregunta = input("\n👤 Tu pregunta: ")
        if pregunta.lower() == 'salir': break

        # k=15 para que la IA tenga mucha "memoria" de los gastos similares
        docs_relevantes = vectorstore.similarity_search(pregunta, k=15)
        contexto_final = "\n\n".join([doc.page_content for doc in docs_relevantes])

        # Prompt optimizado para evitar agrupaciones (como el x4 YPF)
        prompt = f"""
        ERES UN AUDITOR CONTABLE DETALLISTA.
        REGLAS:
        1. NO RESUMAS. Si hay 10 gastos iguales, lístalos los 10 con su fecha y monto.
        2. Mantén un formato limpio y profesional.
        3. Si te pido totales, suma cada ítem individualmente para verificar.

        CONTEXTO EXTRAÍDO:
        {contexto_final}

        PREGUNTA: {pregunta}
        """
        
        response = llm.invoke(prompt)

        # --- LIMPIEZA DE LA RESPUESTA ---
        if hasattr(response, 'content'):
            if isinstance(response.content, list):
                # Extrae solo el texto si Google envía el formato de lista/firmas
                respuesta_limpia = "".join([bloque['text'] for bloque in response.content if 'text' in bloque])
            else:
                respuesta_limpia = response.content
        else:
            respuesta_limpia = str(response)

        print(f"\n🤖 IA:\n{respuesta_limpia}")

    except Exception as e:
        print(f"\n❌ Error: {e}")