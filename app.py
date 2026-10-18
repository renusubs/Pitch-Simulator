import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from strands import Agent
from strands.models import BedrockModel

# Page setup
st.set_page_config(page_title="VC Pitch Evaluator", page_icon="🚀", layout="centered")

st.title("🚀 VC Pitch Evaluator")
st.write("Powered by Amazon Bedrock & Strands SDK")

@st.cache_resource
def load_agent():
    """Initializes and caches the agent so it doesn't reload on every button click."""
    model = BedrockModel(
        model_id="us.amazon.nova-pro-v1:0",
        region_name="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        streaming=False
    )
    return Agent(
        name="StartupEvaluator",
        model=model,
        system_prompt=(
            "You are a lead VC partner evaluating startup pitches. "
            "Perform a 3-part comprehensive evaluation:\n"
            "1. MARKET ANALYSIS: Analyze competitors, target market size, and 2 key differentiators.\n"
            "2. FINANCIAL MODEL: Estimate unit economics, platform take rate (%), and Year 1 GMV.\n"
            "3. VC RISK EVALUATION: Identify top 3 critical risks and score total risk from 1 to 10.\n\n"
            "Format your response clearly using Markdown headings, bullet points, and clean formatting."
        )
    )

agent = load_agent()

# Input area
startup_pitch = st.text_area(
    "Enter your startup idea / pitch:",
    value="BarkDash: An on-demand dog walking and pet grooming mobile app in Austin, TX.",
    height=100
)

if st.button("Evaluate Startup Pitch", type="primary"):
    if not startup_pitch.strip():
        st.warning("Please enter a pitch to evaluate.")
    else:
        with st.spinner("Analyzing market, financial modeling, and VC risk..."):
            try:
                # Call the agent
                result = agent(f"Evaluate startup idea: {startup_pitch}")
                
                st.subheader("📊 Evaluation Report")
                st.markdown(result)
            except Exception as e:
                st.error(f"Error executing agent: {e}")
