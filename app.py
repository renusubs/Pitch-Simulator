import os
import json
import streamlit as st
from dotenv import load_dotenv
import boto3

# Load environment variables securely from .env
load_dotenv(override=True)

# Page setup
st.set_page_config(page_title="VC Pitch Evaluator", page_icon="🚀", layout="centered")

st.title("🚀 VC Pitch Evaluator")
st.write("Powered by Amazon Bedrock & Llama 3")

@st.cache_resource
def get_bedrock_client():
    """Initializes the Bedrock Runtime client using secure local credentials."""
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

client = get_bedrock_client()

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
                prompt = (
                    "You are a lead VC partner evaluating startup pitches. "
                    "Perform a 3-part comprehensive evaluation:\n"
                    "1. MARKET ANALYSIS: Analyze competitors, target market size, and 2 key differentiators.\n"
                    "2. FINANCIAL MODEL: Estimate unit economics, platform take rate (%), and Year 1 GMV.\n"
                    "3. VC RISK EVALUATION: Identify top 3 critical risks and score total risk from 1 to 10.\n\n"
                    f"Startup Idea: {startup_pitch}\n\n"
                    "Format your response clearly using Markdown headings, bullet points, and clean formatting."
                )
                
                # Native Llama 3 payload for invoke_model (avoids converse API colon routing bug)
                body = json.dumps({
                    "prompt": prompt,
                    "max_gen_len": 2048,
                    "temperature": 0.7,
                    "top_p": 0.9
                })
                
                response = client.invoke_model(
                    modelId="us.meta.llama3-3-70b-instruct-v1:0",
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
                
                response_body = json.loads(response.get("body").read())
                result = response_body.get("generation", "")
                
                st.subheader("📊 Evaluation Report")
                st.markdown(result)
            except Exception as e:
                st.error(f"Error executing agent: {e}")
