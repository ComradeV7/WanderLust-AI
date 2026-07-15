import streamlit as st
import requests
import json
import os

# CONFIGURATION
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(
    page_title="Wanderlust AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
    <style>
    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Style the main title */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #2C3E50;
    }
    /* Card style for the itinerary */
    .itinerary-card {
        background-color: #2C3E50;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #FF4B4B;
    }
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: grey;
    }
    /* Custom button styling */
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF1C1C;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Wanderlust AI")
st.markdown("#### *Your Intelligent Travel Consultant*")
st.markdown("---")

# SIDEBAR INPUTS
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=80)
    st.header("Trip Details")
    
    destination = st.text_input("Destination", "Kyoto, Japan", placeholder="e.g., Paris, London, Goa")
    duration = st.slider("Duration (Days)", 1, 14, 3)
    vibe = st.text_area("Vibe / Interests", "Ancient, peaceful, zen gardens, hidden gems", height=100)
    avoid = st.text_input("Avoid (Optional)", placeholder="e.g., Tourist traps, crowded malls")
    
    st.markdown("---")
    
    generate_btn = st.button("Plan My Trip", use_container_width=True)

# APP LOGIC

# Initialize session state for storing data across reloads
if "itinerary" not in st.session_state:
    st.session_state["itinerary"] = ""
if "current_state" not in st.session_state:
    st.session_state["current_state"] = {}

# 1. HANDLE "START PLAN"
if generate_btn:
    with st.spinner(f"Agents are researching {destination}... This may take 30-60 seconds."):
        payload = {
            "destination": destination,
            "duration_days": duration,
            "vibe": vibe,
            "places_to_avoid": [x.strip() for x in avoid.split(",") if x.strip()]
        }
        
        try:
            # Test backend connection first
            try:
                health_response = requests.get(f"{API_URL}/health", timeout=10)
                if health_response.status_code != 200:
                    st.error(f"⚠️ Backend is not responding correctly. Status: {health_response.status_code}")
                    st.info(f"Backend URL: {API_URL}")
                    st.stop()
            except requests.exceptions.RequestException as health_error:
                st.error(f"❌ Cannot connect to backend at: {API_URL}")
                st.error(f"Error: {str(health_error)}")
                st.info("Make sure the backend is deployed and the API_URL environment variable is correct.")
                st.stop()
            
            # Make the actual request
            response = requests.post(f"{API_URL}/plan/start", json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                st.session_state["itinerary"] = data["itinerary_draft"]
                st.session_state["current_state"] = data["current_state"]
                st.rerun() # Refresh to show the itinerary
            else:
                st.error(f"❌ Backend Error (Status {response.status_code})")
                st.error(f"Response: {response.text}")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. This might be a cold start (first request takes ~30s). Please try again.")
        except Exception as e:
            st.error(f"❌ Connection Error: {str(e)}")
            st.info(f"Backend URL: {API_URL}")
            st.info("Check that the backend is running and the URL is correct.")

# DISPLAY ITINERARY
if st.session_state["itinerary"]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="itinerary-card">', unsafe_allow_html=True)
        st.markdown(st.session_state["itinerary"])
        st.markdown('</div>', unsafe_allow_html=True)

    # EEDBACK LOOP (Right Column)
    with col2:
        st.info(" **Refine your plan**")
        st.markdown("Chat with the agent to tweak the itinerary.")
        
        feedback_text = st.text_area("Your Feedback", placeholder="e.g., I don't like museums, give me more food spots!")
        avoid_new = st.text_input("Block a specific place?", placeholder="Name of place to remove")
        
        if st.button("Update Itinerary"):
            with st.spinner("Re-planning based on your feedback..."):
                resume_payload = {
                    "current_state": st.session_state["current_state"],
                    "user_feedback": feedback_text,
                    "place_to_avoid": avoid_new if avoid_new else None
                }
                
                try:
                    response = requests.post(f"{API_URL}/plan/resume", json=resume_payload, timeout=120)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["itinerary"] = data["itinerary_draft"]
                        st.session_state["current_state"] = data["current_state"]
                        st.success("✅ Plan updated!")
                        st.rerun()
                    else:
                        st.error(f"❌ Error (Status {response.status_code}): {response.text}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Connection Error: {str(e)}")

else:
    # Empty State (Welcome Message)
    st.info("Enter your trip details in the sidebar to get started!")
    