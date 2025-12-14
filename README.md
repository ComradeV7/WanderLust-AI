# Wanderlust AI: Travel Planner Agent

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)

**Wanderlust AI** is a stateful, multi-agent travel planner that adapts its search strategy based on the destination's scale. Unlike standard travel bots that hallucinate places or treat every city the same, Wanderlust AI verifies locations using real-world geospatial data and adjusts its scope for Megacities versus Regional Hubs.

<img width="1913" height="944" alt="image" src="https://github.com/user-attachments/assets/e4050aaf-f8c8-4203-8947-99ea61536662" />

---

## The Problem: The "Planning Paralysis"
We all love to travel, but we hate the planning. The modern travel experience is broken:
* **Information Overload:** Planning a simple weekend trip often involves juggling 20+ browser tabs—maps, blogs, reviews, and booking sites.
* **Generic Suggestions:** Most tools offer cookie-cutter "Top 10" lists that don't match your personal style.
* **The "Vibe" Gap:** Search engines are bad at understanding abstract desires. If you ask for a *"spooky, ancient, literary vibe,"* traditional tools fail to translate that feeling into a concrete, logistical plan.

Travelers don't just need a list of coordinates; they need a **concierge** that understands context, filters out the noise, and handles the logistics.

## The Solution: A Neuro-Symbolic Concierge
**Wanderlust AI** is a next-generation travel consultant that bridges the gap between **creative inspiration** and **logistical reality**.

Instead of just searching for keywords, it acts as a reasoning engine:
1.  **It Listens:** It uses **Llama 3** to deeply understand your specific "vibe" and constraints.
2.  **It Verifies:** It uses **Gemini** and **Geospatial Tools** to ensure every suggestion is open, real, and geographically relevant.
3.  **It Adapts:** Whether you are visiting a dense megacity like Tokyo or a coastal town like Visakhapatnam, the agent dynamically adjusts its search strategy to find the hidden gems that matter to *you*.

---

## Key Features

### Dynamic City Awareness
The agent detects the scale of the city via its "Importance Score" and adjusts its search radius automatically:
* **Megacities (e.g., London, NYC):** Radius tightens to **30km** to focus on neighborhoods.
* **Regional Hubs (e.g., Visakhapatnam, Bath):** Radius expands to **200km** to verify day-trip adventures (like caves or hill stations).

### Resilience & "Noun Strategy"
If specific business names cannot be found (common in developing regions), the agent pivots strategies. It switches from searching for specific entities (e.g., "The Grand Hotel") to searching for generic nouns (e.g., "Beach", "Museum") to ensure the user always gets a valid, verified itinerary.

### Human-in-the-Loop (Long-Running State)
The system is stateful. It pauses after generating a draft, allowing the user to provide feedback (e.g., *"I hate museums, give me food spots"*). The agent resumes the graph with the full context history to refine the plan.

---

## Architecture

Wanderlust AI uses a multi-model architecture to solve this:
* **The Brain (Reasoning):** **Llama 3.3 70B** (via Groq) handles cultural nuance, vibe interpretation, and itinerary synthesis.
* **The Hands (Tools):** **Gemini 2.5 Flash** (via Google) handles high-speed, high-volume tool calling and data structuring.
* **The Map (Ground Truth):** **Nominatim (OpenStreetMap)** & **OpenRouteService** provide geospatial verification and routing.

The system is built on **LangGraph** with a sequential flow:

1.  **Vibe Interpreter Agent (Llama 3):** Translates user request ("Spooky, ancient") into strategic search keywords.
2.  **Search Agent (Gemini 2.5):** Takes keywords and attempts to verify them using the **Geospatial Tool** (Nominatim).
3.  **Itinerary Agent (Llama 3):** Synthesizes verified data into a structured markdown plan.

### Deployment Stack
* **Backend:** FastAPI (exposes `/plan/start` and `/plan/resume`).
* **Frontend:** Streamlit (Clean, minimal UI).
* **Containerization:** Docker (Ready for Google Cloud Run).

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ComradeV7/WanderLust-AI.git
```

### 2. Set Environment Variables
```bash
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
ORS_API_KEY=your_openrouteservice_key
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the App

### Terminal 1: Backend (FastAPI)

```bash
uvicorn main:api --reload
```

### Terminal 2: Frontend (Streamlit)

```bash
streamlit run frontend.py
```

---

## Result

<img width="1897" height="902" alt="image" src="https://github.com/user-attachments/assets/b4f8134e-ac4d-4f57-bbc8-46f3e150243b" />


