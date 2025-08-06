# chatbot.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from dotenv import load_dotenv
import gradio as gr


load_dotenv()

# === CONFIGURATION ===
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  
MODEL_PATH = "./models/phi-1_5-Q4_K_M.gguf"
NUM_RESULTS = 5

# === INITIALISATION DES COMPOSANTS ===
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

retriever = vector_store.as_retriever(search_kwargs={'k': NUM_RESULTS})

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.5,
    max_tokens=512,
    top_p=0.95,
    n_ctx=2048,
    verbose=True,
)


# === FONCTION DE STREAMING RÉPONSE ===
def stream_response(message, history):
    # Récupération des documents similaires
    docs = retriever.invoke(message)

    # Affichage des documents pour debug
    print("\n=== DOCUMENTS RETRIEVED ===")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---\n{doc.page_content[:300]}...\n")

    # Fusionner les contenus
    knowledge = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())

    # Vérification s’il y a quelque chose d’utilisable
    if not knowledge.strip():
        yield "I don’t know based on the data I have."
        return

# Prompt structuré
    rag_prompt = f"""<<INSTRUCTIONS>>
You are a career advisor. Recommend ONLY careers that are relevant to the user question and exists in the real world .
You must answer the user question Based EXCLUSIVELY on the knowledge below . 
You must suggest 3-5 careers . 
FORMAT YOUR ANSWER LIKE THIS:

### Recommended Careers:
1. **Career Name**  
   - What They Do: [1 sentence]
   - Why it fits: [1 sentence]  
   - Key skills: [3-5 skills]  
   - Education: [requirements]
   - Salary [range]

2. **Career Name**  
   - What They Do: [1 sentence]
   - Why it fits: [1 sentence]  
   - Key skills: [3-5 skills]  
   - Education: [requirements]
   - Salary [range]

If no matches exist, say: "I couldn't find exact matches, but these might interest you: [list 2-3 closest jobs]"
If NO suitable careers exist or question contradicts reality:
   - Say: "I couldn't find matching careers."
   
<NOTES>
- Never mention unrelated jobs
- Never invent facts
- If knowledge contradicts the question, say "I couldn't find exact matches"
</INSTRUCTIONS>

KNOWLEDGE:
{knowledge}

USER QUESTION: {message}

ANSWER:"""


    print("\n=== PROMPT ENVOYÉ AU LLM ===\n", rag_prompt)

    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response
        yield partial_message


#=== GRADIO CHAT UI ===
chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Describe your skills or ask for a career recommendation...",
        container=False,  
        autoscroll=True,
        scale=7,
    ),
    title=" Career Advisor Pro",
    description="Get personalized career recommendations based on your skills and interests.",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue"
    ),
   
)

# === LANCEMENT ===
if __name__ == "__main__":
    chatbot.launch()

     