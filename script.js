const programs = {
    1: {
        name: "Word Embeddings and Analogies",
        description: "Demonstrates word embeddings using GloVe model to perform vector arithmetic and find word analogies.",
        code: `from gensim.downloader import load
# Load the pre-trained GloVe model (50 dimensions)
print("Loading pre-trained GloVe model (50 dimensions)...")
model = load("glove-wiki-gigaword-50")
# Function to perform vector arithmetic and analyze relationships
def ewr():
    result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
    print("\\nking - man + woman = ?", result[0][0])
    print("similarity:", result[0][1])
    result = model.most_similar(positive=['paris', 'italy'], negative=['france'], topn=1)
    print("\\nparis - france + italy = ?", result[0][0])
    print("similarity:", result[0][1])
    # Example 3: Find analogies for programming
    result = model.most_similar(positive=['programming'], topn=5)
    print("\\nTop 5 words similar to 'programming':")
    for word, similarity in result:
        print(word, similarity)
ewr()`
    },
    2: {
        name: "Word Embeddings Visualization",
        description: "Visualizes word embeddings using PCA dimensionality reduction.",
        code: `import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.downloader import load

# Dimensionality reduction using PCA
def rd(ems):
    pca = PCA(n_components=2)
    r = pca.fit_transform(ems)
    return r

# Visualize word embeddings
def visualize(words, ems):
    plt.figure(figsize=(10, 6))
    for i, word in enumerate(words):
        x, y = ems[i]
        plt.scatter(x, y, marker='o', color='blue')
        plt.text(x + 0.02, y + 0.02, word, fontsize=12)
    plt.show()

# Generate semantically similar words
def gsm(word):
    sw = model.most_similar(word, topn=5)
    for word, s in sw:
        print(word, s)

# Load pre-trained GloVe model from Gensim API
print("Loading pre-trained GloVe model (50 dimensions)...")
model = load("glove-wiki-gigaword-50")

words = ['football', 'basketball', 'soccer', 'tennis', 'cricket']
ems = [model[word] for word in words]
e = rd(ems)
visualize(words, e)
gsm("programming")`
    },
    3: {
        name: "Custom Word2Vec Model",
        description: "Demonstrates training a custom Word2Vec model on domain-specific data.",
        code: `from gensim.models import Word2Vec

# Custom Word2Vec model
def cw(corpus):
    model = Word2Vec(
        sentences=corpus,
        vector_size=50,  # Dimensionality of word vectors
        window=5,        # Context window size
        min_count=1,     # Minimum frequency for a word to be considered
        workers=4,       # Number of worker threads
        epochs=10,       # Number of training epochs
    )
    return model

# Analyze trained embeddings
def anal(model, word):
    sw = model.wv.most_similar(word, topn=5)
    for w, s in sw:
        print(w, s)

# Example domain-specific dataset (medical/legal/etc.)
corpus = [
    "The patient was prescribed antibiotics to treat the infection.".split(),
    "The court ruled in favor of the defendant after reviewing the evidence.".split(),
    "Diagnosis of diabetes mellitus requires specific blood tests.".split(),
    "The legal contract must be signed in the presence of a witness.".split(),
    "Symptoms of the disease include fever, cough, and fatigue.".split(),
]

model = cw(corpus)
print("Analysis for word patient")
anal(model, "patient")
print("Analysis for word court")
anal(model, "court")`
    },
    4: {
        name: "Text Generation with GPT-2",
        description: "Demonstrates text generation using GPT-2 with enriched prompts.",
        code: `from gensim.downloader import load
import torch
from transformers import pipeline

# Load pre-trained word embeddings (GloVe)
model = load("glove-wiki-gigaword-50")
torch.manual_seed(42)

# Define contextually relevant word enrichment
def enrich(prompt):
    ep = ""
    words = prompt.split()
    for word in words:
        sw = model.most_similar(word, topn=3)
        print("Test Data\\n",sw)
        enw=[]
        for s,w in sw:
            enw.append(s)
        ep+=" " + " ".join(enw)
    return ep

# Example prompt to be enriched
op = "lung cancer"
ep = enrich(op)

# Display the results
print("Original Prompt:", op)
print("Enriched Prompt:", ep)
generator = pipeline("text-generation", model="gpt2")
response = generator(op, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
print("\\n\\nPrompt response\\n",response[0]["generated_text"])
response = generator(ep, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
print("\\n\\nEnriched prompt response\\n",response[0]["generated_text"])`
    },
    5: {
        name: "Paragraph Generation",
        description: "Generates meaningful paragraphs using word embeddings.",
        code: `from gensim.downloader import load
import random

# Load the pre-trained GloVe model
print("Loading pre-trained GloVe model (50 dimensions)...")
model = load("glove-wiki-gigaword-50")
print(model)
print("Model loaded successfully!")

# Function to construct a meaningful paragraph
def create_paragraph(iw, sws):
    paragraph = f"The topic of {iw} is fascinating, often linked to terms like\\n"
    random.shuffle(sws)  # Shuffle to add variety
    for word in sws:
        paragraph += str(word) + ", "
    paragraph = paragraph.rstrip(", ") + "."
    return paragraph

iw = "cricket"
sws = model.most_similar(iw, topn=50)
words = [word for word, s in sws]
paragraph = create_paragraph(iw, words)
print(paragraph)`
    },
    6: {
        name: "Sentiment Analysis",
        description: "Performs sentiment analysis on customer feedback using DistilBERT.",
        code: `from transformers import pipeline

# Specify the model explicitly
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

customer_feedback = [
    "The product is amazing! I love it!",
    "Terrible service, I am very disappointed.",
    "This is a great experience, I will buy again.",
    "Worst purchase I've ever made. Completely dissatisfied.",
    "I'm happy with the quality, but the delivery was delayed."
]

for feedback in customer_feedback:
    sentiment_result = sentiment_analyzer(feedback)
    sentiment_label = sentiment_result[0]['label']
    sentiment_score = sentiment_result[0]['score']
    
    # Display sentiment results
    print(f"Feedback is: {feedback}")
    print(f"Sentiment is: {sentiment_label} (Confidence: {sentiment_score:.2f})\\n")`
    },
    7: {
        name: "Text Summarization",
        description: "Summarizes text using BART model.",
        code: `from transformers import pipeline

# Specify the model explicitly
summarizer = pipeline(
    "summarization", 
    model="facebook/bart-large-cnn"
)

# Function to summarize a given passage
def summarize_text(text):
    # Summarizing the text using the pipeline
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

text = """
Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. 
The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is valuable. 
NLP techniques are used in many applications, such as speech recognition, sentiment analysis, machine translation, and chatbot functionality. 
Machine learning algorithms play a significant role in NLP, as they help computers to learn from vast amounts of language data and improve their ability to process and generate text. 
However, NLP still faces many challenges, such as handling ambiguity, understanding context, and processing complex linguistic structures. 
Advances in NLP have been driven by deep learning models, such as transformers, which have significantly improved the performance of many NLP tasks.
"""

# Get the summarized text
summarized_text = summarize_text(text)

# Display the summarized text
print("Original Text:\\n", text)
print("\\nSummarized Text:\\n", summarized_text)`
    },
    8: {
        name: "Institution Information Fetcher",
        description: "Fetches and extracts institution details from Wikipedia.",
        code: `from pydantic import BaseModel
import wikipediaapi

# Define the Pydantic Schema
class InstitutionDetails(BaseModel):
    name: str
    founder: str
    founded: str
    branches: str
    employees: str
    summary: str

# Helper function to extract info based on keyword
def extract_info(content, keyword):
    for line in content.split('\\n'):
        if keyword in line.lower():
            return line.strip()
    return "Not available"

# Function to Fetch and Extract Details from Wikipedia
def fetch(institution_name):
    user_agent = "InstitutionInfoFetcher/1.0"
    wiki = wikipediaapi.Wikipedia('en', headers={"User-Agent": user_agent})
    page = wiki.page(institution_name)

    if not page.exists():
        raise ValueError(f"No Wikipedia page found for '{institution_name}'")

    content = page.text

    founder = extract_info(content, "founder")
    founded = extract_info(content, "founded") or extract_info(content, "established")
    branches = extract_info(content, "branch")
    employees = extract_info(content, "employee")
    summary = "\\n".join(content.split('\\n')[:4])

    return InstitutionDetails(
        name=institution_name,
        founder=founder,
        founded=founded,
        branches=branches,
        employees=employees,
        summary=summary
    )

# Run the program
details = fetch("PESITM")
print("\\nExtracted Institution Details:")
print(details.model_dump_json(indent=4))`
    },
    9: {
        name: "IPC Document Chatbot",
        description: "A chatbot that answers questions about IPC documents.",
        code: `import fitz  # PyMuPDF

# Step 1: Extract Text from IPC PDF
def extract(file):
    text = ""
    with fitz.open(file) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Step 2: Search for Relevant Sections in IPC
def search(query, ipc):
    query = query.lower()
    lines = ipc.split("\\n")
    results=[]
    for line in lines:
        if query in line.lower():
           results.append(line) 
    return results if results else ["No relevant section found."]

# Step 3: Main Chatbot Function
def chatbot():
    print("Loading IPC document...")
    ipc = extract("IPC.pdf")
    while True:
        query = input("Ask a question about the IPC: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        results = search(query, ipc)
        print("\\n".join(results))
        print("-" * 50)`
    }
};

let currentProgram = 1;

function updateProgram() {
    const program = programs[currentProgram];
    document.getElementById('program-title').textContent = `Program ${currentProgram}/9`;
    document.getElementById('program-name').textContent = program.name;
    document.getElementById('program-description').textContent = program.description;
    document.getElementById('program-code').textContent = program.code;
    
    document.getElementById('prev-btn').disabled = currentProgram === 1;
    document.getElementById('next-btn').disabled = currentProgram === 9;
    
    // Refresh syntax highlighting
    Prism.highlightElement(document.getElementById('program-code'));
}

function previousProgram() {
    if (currentProgram > 1) {
        currentProgram--;
        updateProgram();
    }
}

function nextProgram() {
    if (currentProgram < 9) {
        currentProgram++;
        updateProgram();
    }
}

// Copy functionality
async function copyCode() {
    const codeElement = document.getElementById('program-code');
    const code = codeElement.textContent;

    try {
        await navigator.clipboard.writeText(code);
        showToast();
    } catch (err) {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = code;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showToast();
    }
}

// Toast notification
function showToast() {
    const toast = document.getElementById('toast');
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 2000);
}

// Initialize the first program
updateProgram(); 