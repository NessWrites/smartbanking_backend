# Standard Library Imports
import json
from decimal import Decimal, InvalidOperation
import re

# Django Imports
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils.dateparse import parse_date
from django.views.decorators.csrf import csrf_exempt

# Django REST Framework Imports
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView


from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
# Third-Party Imports
from rest_framework_simplejwt.tokens import RefreshToken
import joblib
import numpy as np
import pandas as pd
from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
#from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
#from langchain.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
from llama_cpp import Llama
import torch
#from langchain_core.runnables import Runnable 

# Local Imports
from .models import User, AccountType, Transactions, TransactionType, Account, Withdraw
from .serializers import (
    UserSerializer, AccountTypeSerializer, TransactionsSerializer, TransactionTypeSerializer
)
from .tracker import Tracker

# 1. Create User (Superadmin Only)
class CreateUserView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()  # No need to manually add balance here; it's already set to 500 in the model.
            return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# 2. Login View
class LoginView(APIView):
    def post(self, request):
        phoneNumber = request.data.get('username')
        password = request.data.get('password')
        try:
            user = User.objects.get(phoneNumber = phoneNumber)
            if user.check_password(password):
                refresh = RefreshToken.for_user(user)
                return Response({
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                    # 'user': {
                    #     'id': user.id,
                    #     'username': user.username,
                    #     'email': user.email,
                    #     'phone': user.phone,
                    #     'balance': str(user.account_balance)  # Include balance in response
                    # }
                })
            return Response({"message": "Invalid password"}, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response({"message": "User not found"}, status=status.HTTP_400_BAD_REQUEST)

# 3. User Info View (Authenticated Users Only)
class UserInfoView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

# 4. Refresh Token View
class RefreshTokenView(APIView):
    def post(self, request):
        refresh_token = request.data.get("refresh")
        if not refresh_token:
            return Response({"error": "Refresh token is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            refresh = RefreshToken(refresh_token)
            return Response({"access": str(refresh.access_token)})
        except Exception:
            return Response({"error": "Invalid refresh token"}, status=status.HTTP_400_BAD_REQUEST)

# 5. Check Account Balance
class CheckBalanceView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response({"balance": str(request.user.account_balance)})

# 6. Deposit Money
class DepositView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        amount = request.data.get("amount")
        
        # Validate amount
        try:
            # Ensure amount is a valid decimal and greater than 0
            amount = Decimal(amount)
            if amount <= 0:
                return Response({"error": "Amount must be greater than zero."}, status=status.HTTP_400_BAD_REQUEST)
        except (ValueError, TypeError, Decimal.InvalidOperation):
            return Response({"error": "Invalid amount format."}, status=status.HTTP_400_BAD_REQUEST)

        # Ensure amount is not None or empty
        if amount is None:
            return Response({"error": "Amount is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Get user's account
        try:
            account = Account.objects.get(user=request.user)
            print(account.accountNumber)  # Debug print (you can remove this later)
        except Account.DoesNotExist:
            return Response({"error": "Account not found for this user."}, status=status.HTTP_404_NOT_FOUND)
        
        # Update account balance
        account.balance += amount  # Using Decimal
        account.save()

        # Create transaction record
        try:
            transaction_type = TransactionType.objects.get(transactionType="deposit")
        except TransactionType.DoesNotExist:
            return Response({"error": "Transaction type 'deposit' not found."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        Transactions.objects.create(
            accountID=account,
            transactionTypeID=transaction_type,
            amount=amount,
            reference=f"Deposit {account.accountNumber}",
            description=f"Deposited {amount} to account"
        )

        # Return success response
        return Response({
            "message": f"Deposited {amount} successfully",
            "new_balance": str(account.balance)  # return updated balance
        }, status=status.HTTP_200_OK)


class WithdrawView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        amount = request.data.get("amount")
        withdrawal_type = request.data.get("withdrawal_type")  # 'atm' or 'online_order'
        location = request.data.get("location","Kathmandu")  # ATM location or retailer address
        retailer = request.data.get("retailer") if withdrawal_type == "online_order" else None
        used_chip = request.data.get("used_chip", False)  # Optional, for ATM
        otp_code = request.data.get("otp_code")  # OTP entered by the user (if required)
        repeat_retailer = 0



        # Validate amount
        try:
            amount = Decimal(amount)
            if amount <= 0:
                return Response({"error": "Amount must be greater than zero."}, status=status.HTTP_400_BAD_REQUEST)
        except (ValueError, TypeError, InvalidOperation):
            return Response({"error": "Invalid amount format."}, status=status.HTTP_400_BAD_REQUEST)

        if withdrawal_type not in ["atm", "online_order"]:
            return Response({"error": "Invalid withdrawal type. Must be 'atm' or 'online_order'."}, status=status.HTTP_400_BAD_REQUEST)

        # ATM withdrawal limit
        if withdrawal_type == "atm" and amount > 100000:
            return Response({"error": "ATM withdrawal limit exceeded. Max limit is 100,000."}, status=status.HTTP_400_BAD_REQUEST)

        # Account check
        try:
            account = Account.objects.get(user=request.user)
        except Account.DoesNotExist:
            return Response({"error": "Account not found."}, status=status.HTTP_404_NOT_FOUND)

        if account.balance < amount:
            return Response({"error": "Insufficient balance."}, status=status.HTTP_400_BAD_REQUEST)
        # Get the last location from the most recent withdrawal
        last_withdrawal = Withdraw.objects.filter(user=request.user).order_by('-created_at').first()
        last_location = last_withdrawal.location if last_withdrawal else location
        
        # Calculate the ratio_to_median_purchase_price
        median_purchase_price = request.user.median_purchase_price()
        if median_purchase_price != 0:
            ratio_to_median_purchase_price = float(amount) / float(median_purchase_price)
        else:
            ratio_to_median_purchase_price = 0.0  # Ensure it's float even when zero
        
        # Check if the retailer is repeated (for online orders)
        
        if withdrawal_type == "online_order":
            repeat_retailer = Withdraw.objects.filter( user = request.user,
                retailer=retailer,
                withdrawal_type='online_order'
            ).exists()
            

        tracker = Tracker()

        # Check for potential fraud using XGBoost model
        is_fraudulent = tracker.check_for_fraud(request.user, amount,last_location, location,repeat_retailer, withdrawal_type,ratio_to_median_purchase_price)

        if is_fraudulent:
            # Send OTP if the transaction seems fraudulent
            
            phone_number = request.user.phoneNumber  # Assuming the user's phone number is stored in the User model
            otp_status = tracker.send_otp(phone_number)

            if otp_status != "pending":
                return Response({"error": "Failed to send OTP."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Require OTP verification
            if not otp_code:
                return Response({
                    "message": "OTP sent to your registered phone number.",
                    "requires_otp": True
                }, status=status.HTTP_200_OK)

            # Verify OTP
            if not tracker.verify_otp(phone_number, otp_code):
                return Response({"error": "Invalid OTP. Transaction canceled."}, 
                                status=status.HTTP_400_BAD_REQUEST)

        # Deduct amount
        account.balance -= amount
        account.save()

        # Transaction creation
        transaction_type = TransactionType.objects.get(transactionType="withdraw")
        transaction = Transactions.objects.create(
            accountID=account,
            transactionTypeID=transaction_type,
            amount=amount,
            reference=f"Withdrawal {account.accountNumber}",
            description=f"Withdrew {amount} from account"
        )

        # Withdraw record creation
        withdraw_record = Withdraw.objects.create(
            transaction=transaction,
            user=request.user,
            withdrawal_type=withdrawal_type,
            location=location,
            retailer=retailer,
            used_chip=used_chip
        )

        # Return response
        return Response({
            "message": f"Withdrew {amount} successfully as {withdrawal_type}.",
            "new_balance": str(account.balance)
        }, status=status.HTTP_200_OK)



# 8. View Account Statement
class AccountStatementView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        start_date = request.query_params.get("start_date")
        end_date = request.query_params.get("end_date")
        transactions = Transactions.objects.filter(accountID__user=request.user).order_by("-date")

        # Apply date filtering if both start_date and end_date are provided
        if start_date and end_date:
            start_date = parse_date(start_date)  # Convert string to date
            end_date = parse_date(end_date)

            if start_date and end_date:
                transactions = transactions.filter(date__range=[start_date, end_date])

        serializer = TransactionsSerializer(transactions, many=True)
        return Response(serializer.data)

# 9. Account Type Management (Admin Only)
class AccountTypeView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        account_types = AccountType.objects.all()
        serializer = AccountTypeSerializer(account_types, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = AccountTypeSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk=None):
        account_type = get_object_or_404(AccountType, pk=pk)
        serializer = AccountTypeSerializer(account_type, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk=None):
        account_type = get_object_or_404(AccountType, pk=pk)
        account_type.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

# 10. Transaction Management (Authenticated Users)
class TransactionView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        transactions = Transactions.objects.filter(user=request.user).order_by("-date")
        serializer = TransactionsSerializer(transactions, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = TransactionsSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk=None):
        transaction = get_object_or_404(Transactions, pk=pk, user=request.user)
        transaction.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

# 11. Transaction Type Management (Admin Only)
class TransactionTypeView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        transaction_types = TransactionType.objects.all()
        serializer = TransactionTypeSerializer(transaction_types, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = TransactionTypeSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk=None):
        transaction_type = get_object_or_404(TransactionType, pk=pk)
        serializer = TransactionTypeSerializer(transaction_type, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk=None):
        transaction_type = get_object_or_404(TransactionType, pk=pk)
        transaction_type.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

# 12. User Management (Admin Only)
class UserView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        users = User.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk=None):
        user = get_object_or_404(User, pk=pk)
        serializer = UserSerializer(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk=None):
        user = get_object_or_404(User, pk=pk)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["query"],
    template=(
        "User: {query}\n"
        "Assistant: Identify the user's intent and provide a **single, complete response**. "
        "Use placeholders like {{Account Name}}, {{Customer Name}}, {{Customer ID}}, {{Customer Support}}, {{Account Balance}}, etc., for dynamic data.\n"
        "Format your response **strictly** as follows:\n"
        "Intent: <intent_here>\n"
        "Response: <detailed_response_here>\n"
        "Ensure the response is complete and does not end abruptly. Do not include any additional text, instructions, or explanations. Only provide the intent and response."
    )
)
#Define a function to fetch user data from the database (this can be improved based on your needs)
def fetch_user_data(user_id):
    try:
        # Fetch user from your Django database
        user = User.objects.get(accountNumber=user_id)
        
        # For example, returning the balance and recent transactions
        user_data = {
            "balance": user.balance,
            "last_transactions": user.get_last_transactions(),
            "account_number": user.accountNumber,
        }
        return user_data
    except User.DoesNotExist:
        return None

# Define a basic retriever using the user data
class SimpleRetriever:
    def __init__(self, user_id):
        self.user_id = user_id
    
    def retrieve(self, query):
        user_data = fetch_user_data(self.user_id)
        
        if user_data is None:
            return f"Sorry, I couldn't find the account information for user {self.user_id}."
        
        # Example retrieval logic: Return user balance if asking for balance
        if "balance" in query.lower():
            return f"Your current balance is {user_data['balance']}."
        
        # You can add more conditions based on the query
        return f"Recent transactions: {user_data['last_transactions']}"

# Load the Llama model (as before)
model_path = "/Users/ness/Islington/unsloth.Q4_K_M.gguf"
llm = LlamaCpp(model_path=model_path)

# # Define the prompt template
# prompt_template = PromptTemplate(
#     input_variables=["query", "retrieved_data"],
#     template="You are a helpful assistant. Here is the user's data: {retrieved_data}. Now, based on the query: {query}, provide a response."
# )

# # Define a new chain that includes both retrieval and language model processing
# def create_chain(user_id):
#     retriever = SimpleRetriever(user_id)
    
#     # Here, the retrieval happens first, then the LLM is used to generate the response
#     chain = LLMChain(llm=llm, prompt=prompt_template)
#     return chain, retriever
# Define the prompt template
# prompt_template = PromptTemplate(
#     input_variables=["query"],
#     template=(
#         "You are a helpful assistant. Provide detailed information about the user based on the following query: {query}\n"
#         "If the query is about account balance or transactions, retrieve the data from the database and include it in the response."
#     )
# )

# Initialize the chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# # Initialize embeddings and vector store (for RAG)
# embeddings = HuggingFaceEmbeddings()
# vector_store = FAISS.load_local("/Users/ness/Islington/smartbanking_backend/smart_banking/users", embeddings,allow_dangerous_deserialization=True)  # Load your pre-built vector store

# # Initialize RetrievalQA chain
# retrieval_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vector_store.as_retriever(),
#     chain_type_kwargs={"prompt": prompt_template}
# )


# Authentication Check
def is_authenticated(user_id):
    try:
        # Check if the user exists in the database
        user = User.objects.get(accountNumber=user_id)
        return True
    except User.DoesNotExist:
        return False
    
# Process Query Function
def process_query(query, user_id):
    if not is_authenticated(user_id):
        # Unauthenticated user: Use the fine-tuned model directly
        output = llm(query, max_tokens=400, temperature=0.7, top_p=0.9)
        if 'choices' in output and output['choices']:
            return output['choices'][0]['text'].strip()
        else:
            return "Sorry, I couldn't generate a response."
    else:
        # Authenticated user: Use LangChain to retrieve and process user-specific data
        try:
            # Fetch user data
            user = User.objects.get(accountNumber=user_id)
            user_data = user.get_user_data()

            # Detect intent based on the query
            intent = detect_intent(query)

            # Handle specific intents
            if intent == "check_balance":
                # Fetch account balance from the database
                balance = user_data.get("Account Balance", "not available")
                response = f"Your account balance is {balance}."
            elif intent == "recent_transactions":
                # Fetch recent transactions from the database
                transactions = user.get_transaction_details()[:5]  # Last 5 transactions
                response = "Your last 5 transactions are:\n"
                for transaction in transactions:
                    response += f"- {transaction['transactionType']}: {transaction['amount']} on {transaction['date']}\n"
            else:
                # Use RAG for other queries
                response = llm.invoke(query)

            return response

        except User.DoesNotExist:
            return "User not found."
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Intent Detection Function
def detect_intent(query):
    """
    Detects the intent of the user's query.
    """
    query = query.lower()

    if "balance" in query or "check balance" in query:
        return "check_balance"
    elif "transaction" in query or "recent transactions" in query:
        return "recent_transactions"
    else:
        return "unknown"
    
# def process_query(query, user_id):
#     if not is_authenticated(user_id):
#         # Unauthenticated user: Use the fine-tuned model directly
#         output = llm(query, max_tokens=300, temperature=0.7, top_p=0.9)
#         if 'choices' in output and output['choices']:
#             return output['choices'][0]['text'].strip()
#         else:
#             return "Sorry, I couldn't generate a response."
#     else:
#         # Authenticated user: Use LangChain to retrieve and process user-specific data
#         try:
#             # Create the retrieval and chain
#             chain, retriever = create_chain(user_id)

#             # Retrieve the data for the user
#             retrieved_data = retriever.retrieve(query)

#             # Generate the response using LangChain
#             response = chain.run({"query": query, "retrieved_data": retrieved_data})
#             return response

#         except User.DoesNotExist:
#             return "User not found."
#         except Exception as e:
#             return f"An error occurred: {str(e)}"

# Clean Response Function
def clean_response(response):
    """
    Cleans the response to ensure only one intent and response are included.
    """
    # Split the response into lines
    lines = response.split("\n")
    
    # Initialize variables to store intent and response
    intent = None
    response_text = None
    
    # Iterate through lines to find the first valid intent and response
    for line in lines:
        if line.startswith("Intent:") and not intent:
            intent = line.strip()
        elif line.startswith("Response:") and not response_text:
            response_text = line.strip()
        if intent and response_text:
            break
    
    # If both intent and response are found, return them
    if intent and response_text:
        return f"{intent}\n{response_text}"
    else:
        return f"Intent: unknown\nResponse: Sorry, I couldn't generate a valid response.\n{response_text}"

# Placeholder Replacement Function
def replace_placeholders(text, user_data):
    """
    Replaces placeholders in the text with actual user data.
    """
    placeholder_pattern = re.compile(r'\{\{([^}]+)\}\}')
    def replace_match(match):
        placeholder = match.group(1).strip()
        value = user_data.get(placeholder, f"{{{{{placeholder}}}}}")
        return str(value)
    return placeholder_pattern.sub(replace_match, text)

# Chat Endpoint
# Chat Endpoint
@csrf_exempt
def chat(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_id = data.get("user_id")
            query = data.get("query")

            if not query or not user_id:
                return JsonResponse({"error": "No query provided or user not logged in"}, status=400)

            response = process_query(query, user_id)
            return JsonResponse({"response": response}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in request"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)