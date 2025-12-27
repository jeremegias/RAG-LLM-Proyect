import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# Esta es la nueva herramienta para leer carpetas de PDFs
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

# 1. CONFIGURACIÓN DEL MODELO (Igual que antes)
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 2. CARGA DE DOCUMENTOS (Aquí está el gran cambio)
# Asegúrate de crear una carpeta llamada 'documentos' y poner tus PDFs ahí
if not os.path.exists("documentos"):
    os.makedirs("documentos")
    print("📁 Carpeta 'documentos' creada. Agrega tus PDFs allí.")

loader = PyPDFDirectoryLoader("documentos/")
documentos_crudos = loader.load()
print(f"📄 Se han cargado {len(documentos_crudos)} páginas de PDFs.")

# 3. FRAGMENTACIÓN (Adaptado para documentos)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
fragmentos = text_splitter.split_documents(documentos_crudos)
print(f"✂️ Documentos divididos en {len(fragmentos)} fragmentos.")

# 4. BASE DE DATOS VECTORIAL
# Usamos 'from_documents' en lugar de 'from_texts' porque el PDF trae metadatos (página, nombre de archivo)
vectorstore = Chroma.from_documents(
    documents=fragmentos, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 5. Bucle de chat interactivo
print("\n--- 🤖 Chat con tus PDFs listo ---")
while True:
    try:
        pregunta = input("\n👤 Tu pregunta (o escribe 'salir'): ")
        if pregunta.lower() == 'salir': break

        # Búsqueda de contexto
        docs_relevantes = vectorstore.similarity_search(pregunta, k=4)
        contexto_final = "\n\n".join([doc.page_content for doc in docs_relevantes])

        # Prompt ultra-directo
        prompt = f"Contexto: {contexto_final}\n\nPregunta: {pregunta}"
        
        # Invocamos al modelo
        response = llm.invoke(prompt)

        # LIMPIEZA TOTAL: LangChain extrae el texto principal en .content
        # Si Gemini envía firmas, aquí solo tomamos el mensaje final.
        if hasattr(response, 'content'):
            # Si es una lista (formato con firmas), extraemos solo el texto
            if isinstance(response.content, list):
                respuesta_limpia = "".join([p['text'] for p in response.content if 'text' in p])
            else:
                respuesta_limpia = response.content
        else:
            respuesta_limpia = str(response)

        print(f"\n🤖 IA: {respuesta_limpia}")

    except Exception as e:
        if "429" in str(e):
            print("\n⚠️ Límite de Google alcanzado. Espera 10 segundos y vuelve a intentar.")
        else:
            print(f"\n❌ Error inesperado: {e}")