import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API if key is available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Instantiate the model (using recommended models for text)
    model = genai.GenerativeModel('gemini-1.5-pro')

def generate_incident_analysis(context_data):
    """
    Generates an incident analysis report using LLM based on traffic context.
    Acts as a placeholder/mock if API key is not provided.
    """
    prompt = f"""
    You are an expert AI traffic safety analyst monitoring a cyber-physical transportation system.
    Analyze the following real-time traffic data and provide an incident analysis report.
    
    Traffic Data Context:
    - Speed: {context_data.get('speed', 'N/A')} km/h
    - Lane Occupancy: {context_data.get('occupancy', 'N/A') * 100:.1f}%
    - Traffic Flow: {context_data.get('flow', 'N/A')} vehicles/hr
    - Weather: {context_data.get('weather', 'N/A')}
    - Sudden Braking Events: {context_data.get('braking', 'N/A')}
    - Predicted Collision Probability: {context_data.get('collision_prob', 'N/A') * 100:.1f}%
    - Predicted Congestion Level: {context_data.get('congestion', 'N/A') * 100:.1f}%
    - Risk Category: {context_data.get('risk', 'N/A')}

    Please provide a brief report with the following structure:
    1. Safety Summary
    2. Hazard Explanation
    3. Root-Cause Analysis
    4. Mitigation Recommendations
    """

    if GEMINI_API_KEY:
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error connecting to Gemini API: {e}\n\nFalling back to simulated analysis..."
    
    # Mock Response
    return mock_llm_response(context_data)

def mock_llm_response(context_data):
    risk = context_data.get('risk', 'Low')
    weather = context_data.get('weather', 'Clear')
    
    if risk == 'High':
        return f"""
### 1. Safety Summary
**High collision risk detected.** Immediate attention required. Predicted collision probability is {context_data.get('collision_prob', 0)*100:.1f}%.

### 2. Hazard Explanation
Dangerous traffic conditions observed due to high lane occupancy ({context_data.get('occupancy', 0)*100:.1f}%) combined with {weather} weather. Multiple sudden braking events ({context_data.get('braking', 0)}) detected.

### 3. Root-Cause Analysis
The primary cause of the heightened risk is the lack of safe following distance in dense traffic, exacerbated by reduced visibility and road traction due to {weather} conditions.

### 4. Mitigation Recommendations
- **Automated Actions:** Activate adaptive speed limits (reduce to 60 km/h).
- **Driver Alerts:** Broadcast hazardous weather and sudden braking warnings to connected vehicles (V2X).
- **Traffic Routing:** Divert upstream traffic to alternative routes to reduce congestion buildup.
        """
    elif risk == 'Medium':
        return f"""
### 1. Safety Summary
**Moderate traffic risk detected.** Congestion is building up, requiring monitoring.

### 2. Hazard Explanation
Traffic flow is increasing, leading to moderate lane occupancy and occasional sudden braking. 

### 3. Root-Cause Analysis
Standard peak hour volume building up. Potential minor bottlenecks forming.

### 4. Mitigation Recommendations
- **Automated Actions:** Optimize traffic signal timing for main arterials.
- **Driver Alerts:** Display "Congestion Ahead" on Variable Message Signs (VMS).
        """
    else:
        return """
### 1. Safety Summary
**Normal traffic conditions.** No immediate safety risks detected.

### 2. Hazard Explanation
Traffic is flowing smoothly with adequate spacing and minimal sudden braking events.

### 3. Root-Cause Analysis
Standard off-peak traffic patterns.

### 4. Mitigation Recommendations
- Maintain current signal timing plans.
- Continue standard monitoring.
        """
