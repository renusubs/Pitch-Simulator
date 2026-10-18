import sys
import os
import logging

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # Or us-west-2

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

from strands import Agent
from strands.models import BedrockModel

print("-> Initializing Llama 3.3 70B model (Non-Streaming)...")
model = BedrockModel(
    model_id="us.meta.llama3-3-70b-instruct-v1:0",
    streaming=False
)

print("-> Initializing Startup Pitch Evaluator Agent...")
# We combine the Market, Financial, and VC Risk expertise into a single multi-persona agent
evaluator_agent = Agent(
    name="StartupEvaluator",
    model=model,
    system_prompt=(
        "You are a lead VC partner evaluating startup pitches. "
        "Perform a 3-part comprehensive evaluation:\n"
        "1. MARKET ANALYSIS: Analyze competitors, target market size, and 2 key differentiators.\n"
        "2. FINANCIAL MODEL: Estimate unit economics, platform take rate (%), and Year 1 GMV.\n"
        "3. VC RISK EVALUATION: Identify top 3 critical risks and score total risk from 1 to 10.\n\n"
        "Return the final analysis in a clean, structured JSON format."
    )
)

if __name__ == "__main__":
    print("-> Starting evaluation execution...")
    prompt = "BarkDash: An on-demand dog walking and pet grooming mobile app in Austin, TX."

    try:
        print(f"-> Sending prompt to agent: '{prompt}'")
        result = evaluator_agent(f"Evaluate startup idea: {prompt}")

        print("\n=== AGENT EVALUATION RESULT ===")
        print(result)
    except Exception as e:
        print(f"\n[ERROR ENCOUNTERED]: {e}")