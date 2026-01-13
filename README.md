# RAG-API
Created a RAG API using FastAPI, ChromaDB to store the knowledge base into embeddings, ollama tu run the llm locally and finally the uvicorn to perform APIs request and respond call more seamlessly. It answers the questions asked by generating them via inferring the knowledge base first for better results.

# Build a RAG API with FastAPI

**Project Link:** [View Project](http://learn.nextwork.org/projects/ai-devops-api)

**Author:** Om Bansod  
**Email:** ombansod39@gmail.com

---

![Image](http://learn.nextwork.org/cheerful_vermilion_radiant_date/uploads/ai-devops-api_g3h4i5j6)

---

## Introducing Today's Project!

In this project, I will demonstrate creation of RAG using Fast API. I'm doing this project to learn the use of knowledge base retrival for AI-powered question-answering system using RAG.

### Key services and concepts

Services I used were FastAPI, ChromaDB, Uvicorn, Ollama. Key concepts I learnt include RAG API creation with dynamic updateable knowledge base, creating interactive API documentation using Swagger UI and testing my API using both curl and Swagger UI.

### Challenges and wins

This project took me approximately 35mins . The most challenging part was adding the dynamic updation part for my knowledge base. It was most rewarding to make updation seamlessly without extra efforts and restarting the server.

### Why I did this project

I did this project because i always have something for AI and thus decided to take step in learning production ready RAG creation now.

---

## Setting Up Python and Ollama

In this step, I'm setting up Python and Ollama. Python is 3.13.5. Ollama is 0.13.5. I need these tools because I need FastAPI framework and ollama to generate AI responses locally.

### Python and Ollama setup

![Image](http://learn.nextwork.org/cheerful_vermilion_radiant_date/uploads/ai-devops-api_i9j0k1l2)

### Verifying Python is working

### Ollama and tinyllama ready

Ollama is 0.13.5. I downloaded the tinyllama model because i need it to run locally efficiently. The model will help my RAG API by generating AI-powered responses through inferring ChromaDB knowledge base .

---

## Setting Up a Python Workspace

In this step, I'm setting up my virtual Python workspace. I need it because it will store all the libs and frameworks or any dependencies i will be installing in that folder venv itself.

### Python workspace setup

### Virtual environment

A virtual environment is a seperate environment used for individual projects. I created one for this project to seperate it from other projects of my local machine. Once I activate it, anything that i will install will be stored for this project only and no other project can access it . To create a virtual environment, I specifically used python -3.13 in command as i require that only rather than -3.14.

### Dependencies

The packages I installed are FastAPI, Uvicorn, ChromaDB, Ollama. FastAPI is used for handing incoming questions. Chroma is used for document embeddings to retrive data faster. Uvicorn is used for a server that runs our FastAPI app and makes it accessible locally on our computer. Ollama is used for generating answers.

![Image](http://learn.nextwork.org/cheerful_vermilion_radiant_date/uploads/ai-devops-api_u1v2w3x4)

---

## Setting Up a Knowledge Base

In this step, I'm creating a Knowledge Base. A knowledge base is like a storage for answers which can be retrived if it matches the question's prompt. I need it because my RAG will retrieve relevant information before getting an answer.

### Knowledge base setup

![Image](http://learn.nextwork.org/cheerful_vermilion_radiant_date/uploads/ai-devops-api_t1u2v3w4)

### Embeddings created

Embeddings are numerical representations of text that capture meaning. I created them by writing python queries in a file and running them. The db/ folder contains stores my knowledge base's embeddings. This is important for RAG because it will allow to search faster and more efficiently when our api is running.

---

## Building the RAG API

In this step, I'm building a RAG API. An API is Application Programming Interface that lets software retrieve and share data with other apps. FastAPI is a high-performance Python web framework for building APIs. I'm creating this because it will my RAG system an API means others can use it beyond my terminal.

### FastAPI setup

### How the RAG API works

My RAG API works by creating a web server with one endpoint /query using FastAPI. 
User will send a queston to my API then chroma will searches through my knowledge base to find text that matches the question's meaning. 
Chroma returns the the most relevant information from my documents.
The question and the matching text are sent together to tinyllama, which creates an answer and the answer is sent back to the user.

![Image](http://learn.nextwork.org/cheerful_vermilion_radiant_date/uploads/ai-devops-api_f3g4h5i6)

---

## Testing the RAG API

In this step, I'm testing my RAG API. I'll test it using command line first and then Swagger UI. Swagger UI is  an automatically generated, interactive documentation page for my FastAPI server. I'll use it to visually explore my API's endpoints, see what parameters they accept, and even try them out right from my browser .

### Testing the API

### API query breakdown

I queried my API by running the command
 {Invoke-RestMethod -Uri "http://127.0.0.1:8000/query?q=What is Kubernetes?" -Method Post}.
The command uses the POST method, which means it sends data to the server to perform some actions. 
The API responded with answer
------
Yes, I can answer the question "What is Kubernetes?" 
Response: Kubernees is a Kubernetes-based container orchestration platform that provides automated management and deployment of containerized applications in cloud e...


![Image](http://learn.nextwork.org/cheerful_vermilion_radiant_date/uploads/ai-devops-api_g3h4i5j6)

### Swagger UI exploration

Swagger UI is an automatically generated, interactive documentation page for your FastAPI server. I used it to test my RAG API by requesting using POST method. The best part about using Swagger UI was it shows :
- All available endpoints (like /query)
- HTTP methods (POST, GET, etc.)
- Required parameters
- Response formats
- A way to test endpoints directly in the browser.
And thus provides a better UI and keeps minimal interaction with the system directly like excluding to remember port number and all syntax etc.

---

## Adding Dynamic Content

In this project extension, I'm going to add a new endpoint to my API that lets me dynamically add content to my knowledge base - the same way production APIs allow users to update data in real-time!.

### Adding the /add endpoint

![Image](http://learn.nextwork.org/cheerful_vermilion_radiant_date/uploads/ai-devops-api_w9x0y1z2)

### Dynamic content endpoint working

The /add endpoint allows me to dynamically extend my knowledge base. This is useful because more my knowledge base gets updated more it will be efficient to acquire accurate information .

---

---
