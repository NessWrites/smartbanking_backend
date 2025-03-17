ai_model.py:
import re
import json
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from llama_cpp import Llama

router = APIRouter()

MODEL_PATH = r"C:\Users\hp\FYP_website\model\unsloth.Q4_K_M.gguf"

# Load the GGUF model
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,  # Context length
        verbose=True  # Enable logging
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

class LoanApplicationRequest(BaseModel):
    customer_data: dict

def query_ai_model(customer_data: dict) -> dict:
    """
    Query the GGUF model and extract a structured JSON response.
    """

    prompt = f"""
    Given the following customer data, assess their loan eligibility and provide an approval decision along with a detailed explanation.
    
    Customer Data:
    {json.dumps(customer_data, indent=2)}

    Respond *ONLY* in the following strict JSON format:
    
⁠     {{
        "Decision": "Approve" or "Reject",
        "Reasoning": "A detailed explanation of the decision."
    }}
     ⁠
    Do not include any extra text outside the JSON format.
    """

    try:
        # Run inference
        response = llm(prompt, max_tokens=200, temperature=0.2)
        response_text = response.get("choices", [{}])[0].get("text", "").strip()

        print(f"Raw AI response: {response_text}")

        # Extract JSON part using regex
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError("Failed to extract JSON from AI response.")

        json_text = match.group(0)
        ai_response = json.loads(json_text)  # Parse JSON

        # Validate response format
        if "Decision" not in ai_response or "Reasoning" not in ai_response:
            raise ValueError("AI response missing required fields.")

        return ai_response

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI model returned an invalid JSON response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI model error: {str(e)}")

@router.post("/loan_application_details/")
async def analyze_loan_application(request: LoanApplicationRequest):
    try:
        print("In AI model routes analyze")
        customer_data = request.customer_data
        print(f"Customer data: {json.dumps(customer_data, indent=2)}")

        # Get AI model response
        ai_response = query_ai_model(customer_data)
        print(f"Parsed AI response: {ai_response}")

        return {
            "aiDecision": ai_response["Decision"],
            "aiReasoning": ai_response["Reasoning"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

from app.routes.ai_model import analyze_loan_application, LoanApplicationRequest
@router.get("/loan_application_details/{loan_id}")
async def get_loan_application_details(loan_id: int, db: Session = Depends(get_db)):
    loan_application = db.query(LoanApplicationData).filter(LoanApplicationData.ReportID == loan_id).first()
    if not loan_application:
        raise HTTPException(status_code=404, detail="Loan application not found")

    user = db.query(User).filter(User.id == loan_application.userID).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    social_info = db.query(SocialInfo).filter(SocialInfo.userID == user.id).first()
    financial_data = db.query(FinancialData).filter(FinancialData.userID == user.id).first()

    customer_data = {
        "age": None,  
        "gender": social_info.gender if social_info else None,
        "marital_status": social_info.maritalStatus if social_info else None,
        "education_level": social_info.educationLevel if social_info else None,
        "occupation": social_info.occupation if social_info else None,
        "income": financial_data.income if financial_data else None,
        "expenses": financial_data.expense if financial_data else None,
        "assets": None,
        "liabilities": financial_data.liability if financial_data else None,
        "employment_status": financial_data.employmentStatus if financial_data else None,
        "loanAmount": loan_application.loanAmount,
        "durationMonths": loan_application.durationMonths,
        "purpose": loan_application.purpose,
        "totalLoans": None,
        "onTimeRepayments": None,
        "delayedRepayments": None,
        "defaults": None,
        "creditScore": None,
        "savingsBalance": financial_data.savingBalance if financial_data else None,
        "familySize": social_info.familySize if social_info else None,
        "oldDependents": social_info.oldDependencies if social_info else None,
        "youngDependents": social_info.youngDependencies if social_info else None,
        "homeOwnership": social_info.homeOwnership if social_info else None,
        "economicSector": loan_application.economicSector
    }

    try:
        print("in the try block of AI model method call")
        customer_data_json = json.dumps(customer_data, default=convert_decimal)
        
        customer_data_dict = json.loads(customer_data_json)
        print(f"Customer data in loan application details: {customer_data_dict}")
        
        # Call AI model properly with ⁠ await ⁠
        ai_response = await analyze_loan_application(LoanApplicationRequest(customer_data=customer_data_dict))

        ai_decision = ai_response.get("aiDecision", "Pending")
        ai_reasoning = ai_response.get("aiReasoning", "No explanation provided.")
    
    except Exception as e:
        print(f"AI Model Error: {str(e)}")
        ai_decision = None
        ai_reasoning = None

    return {
        "age": None,
        "loanId": loan_application.ReportID,
        "firstName": user.first_name,
        "lastName": user.last_name,
        "gender": social_info.gender if social_info else None,
        "marital_status": social_info.maritalStatus if social_info else None,
        "education_level": social_info.educationLevel if social_info else None,
        "occupation": social_info.occupation if social_info else None,
        "income": financial_data.income if financial_data else None,
        "expenses": financial_data.expense if financial_data else None,
        "liabilities": financial_data.liability if financial_data else None,
        "employment_status": financial_data.employmentStatus if financial_data else None,
        "loanAmount": loan_application.loanAmount,
        "durationMonths": loan_application.durationMonths,
        "purpose": loan_application.purpose,
        "savingsBalance": financial_data.savingBalance if financial_data else None,
        "familySize": social_info.familySize if social_info else None,
        "oldDependents": social_info.oldDependencies if social_info else None,
        "youngDependents": social_info.youngDependencies if social_info else None,
        "homeOwnership": social_info.homeOwnership if social_info else None,
        "economicSector": loan_application.economicSector,
        "aiDecision": ai_decision,
        "aiReasoning": ai_reasoning
    }