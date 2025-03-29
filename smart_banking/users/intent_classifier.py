import re
import logging
from typing import Dict, Any, Optional
from enum import Enum
from langchain_community.llms import LlamaCpp
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, validator
from django.db.models import Q
from .models import Account, Loans, Transactions
from .serializers import AccountSerializer, LoansSerializer, TransactionsSerializer
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

class QueryType(str, Enum):
    DIRECT = "direct"
    STEPS = "steps"
    CALCULATIONS = "calculations"

class BankingResponse(BaseModel):
    query: str
    type: QueryType
    response: str
    confidence: float = 1.0
    source: Optional[str] = None
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

class QueryClassifier:
    """Handles query classification with precise pattern matching and LLM fallback"""
    def __init__(self, llm: LlamaCpp):
        self.llm = llm
        self._initialize_prompts()
        
    def _initialize_prompts(self):
        """Initialize classification prompts"""
        self.classifier_prompt = PromptTemplate.from_template("""
            Classify this banking query into exactly one category:
            
            DIRECT - For direct information requests:
            - "my balance", "check my balance", "current balance"
            - "my transactions", "transaction history"
            - "my loan details", "loan status"
            - "account statement"
            
            STEPS - For procedural/how-to questions:
            - "how to check balance", "steps to view transactions"
            - "how to apply for loan", "process to transfer money"
            - "what's needed to open account"
            
            CALCULATIONS - For math/financial calculations:
            - "calculate interest on 5000", "compute EMI for 10000"
            - "convert 200 USD to NPR", "what's 5% of 10000"
            
            Respond with ONLY the classification word: direct, steps, or calculations
            
            Query: {query}
            Classification:
        """)
        
    def classify(self, query: str) -> QueryType:
        """Classify query with precise distinction between direct and steps"""
        try:
            # First try precise pattern matching
            query_type = self._precise_pattern_match(query)
            if query_type:
                return query_type
                
            # Fall back to general pattern matching
            query_type = self._general_pattern_match(query)
            if query_type:
                return query_type
                
            # Finally use LLM classification if patterns don't match
            return self._llm_classify(query)
            
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return QueryType.STEPS
    
    def _precise_pattern_match(self, query: str) -> Optional[QueryType]:
        """Precise pattern matching for clear cases"""
        query_lower = query.lower().strip()
        
        # Explicit STEPS patterns (must come first)
        steps_phrases = [
            'how to', 'steps to', 'way to', 'process to',
            'procedure for', 'guide me', 'show me how',
            'tell me how', 'explain how', 'walk me through',
            'what do i do to', 'what should i do to'
        ]
        if any(phrase in query_lower for phrase in steps_phrases):
            return QueryType.STEPS
            
        # Explicit DIRECT patterns
        direct_phrases = [
            'my balance', 'check balance', 'current balance',
            'my transactions', 'transaction history',
            'my loan', 'loan details', 'loan status',
            'account statement', 'send money', 'transfer money',
            'what is my balance', 'show my balance',
            'view my balance', 'check my account'
        ]
        if any(phrase in query_lower for phrase in direct_phrases):
            return QueryType.DIRECT
            
        return None
    
    def _general_pattern_match(self, query: str) -> Optional[QueryType]:
        """General pattern matching for less clear cases"""
        query_lower = query.lower()
        
        # Calculation patterns
        if (re.search(r"\b(calculate|compute|convert|emi|interest)\b", query_lower) or
            re.search(r"\b\d+\s*%\s+of\s+\d+\b", query_lower)):
            return QueryType.CALCULATIONS
            
        # Steps patterns (weaker signals)
        if (re.search(r"\b(how|steps|process)\s+to\b", query_lower) or
            re.search(r"\b(what('s| is)\s+needed\b", query_lower)):
            return QueryType.STEPS
            
        # Direct patterns (weaker signals)
        if (re.search(r"\b(my|check|view|show)\s+(balance|transactions?|loans?)\b", query_lower) or
            re.search(r"\b(account\s+statement|loan\s+status)\b", query_lower)):
            return QueryType.DIRECT
            
        return None
    
    def _llm_classify(self, query: str) -> QueryType:
        """Use LLM for ambiguous cases"""
        try:
            chain = self.classifier_prompt | self.llm | StrOutputParser()
            raw_output = chain.invoke({"query": query})
            clean_output = self._clean_output(raw_output)
            return QueryType(clean_output)
        except:
            return QueryType.STEPS  # Default fallback
    
    def _clean_output(self, raw: str) -> str:
        """Normalize classifier output"""
        clean = raw.strip().lower()
        clean = re.sub(r"[^a-z]", "", clean)  # Remove non-alphabetic chars
        
        # Handle variations
        if clean.startswith(("dir", "dat", "inf")):
            return "direct"
        if clean.startswith(("step", "proc", "how")):
            return "steps"
        if clean.startswith(("calc", "comp", "math")):
            return "calculations"
            
        return "steps"  # Default fallback

class BankingAssistant:
    def __init__(self, llm: LlamaCpp, user_id: int = None, max_history: int = 10):
        self.llm = llm
        self.user_id = user_id
        self.max_history = max_history
        self.message_history = ChatMessageHistory()
        
        # Initialize components
        self.classifier = QueryClassifier(llm)
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent()
    
    def _initialize_tools(self) -> list:
        return [
            Tool(
                name="AccountLookup",
                func=self._fetch_from_database,
                description="Useful for account balance, transactions, and loan details"
            ),
            Tool(
                name="FinancialCalculator",
                func=self._financial_calculator,
                description="Useful for interest, EMI, and financial calculations"
            ),
            Tool(
                name="CurrencyConverter",
                func=self._currency_converter,
                description="Useful for converting between currencies"
            )
        ]
    
    def _initialize_agent(self):
        memory = ConversationBufferMemory(
            chat_memory=self.message_history,
            memory_key="chat_history",
            return_messages=True
        )
        return initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
    def _fetch_from_database(self, query: str) -> str:
        """Handle authenticated database queries with robust error handling"""
        if not self.user_id:
            return "Please log in to access your account information."
        
        try:
            account = Account.objects.get(user__id=self.user_id)
            query_lower = query.lower()
            
            if "balance" in query_lower:
                try:
                    # Ensure balance is converted to float before formatting
                    balance = float(AccountSerializer(account).data['balance'])
                    return f"Current balance: NPR {balance:,.2f}"
                except (ValueError, TypeError, KeyError) as e:
                    logger.error(f"Balance formatting error: {str(e)}")
                    return "Could not retrieve balance information."
                
            elif "transaction" in query_lower:
                try:
                    transactions = Transactions.objects.filter(
                        Q(account=account) & Q(status='completed')
                    ).order_by('-date')[:5]
                    return self._format_transactions(transactions)
                except Exception as e:
                    logger.error(f"Transaction fetch error: {str(e)}")
                    return "Could not retrieve transaction history."
                
            elif "loan" in query_lower:
                try:
                    return self._get_loan_details(account)
                except Exception as e:
                    logger.error(f"Loan details error: {str(e)}")
                    return "Could not retrieve loan information."
                
            return "I couldn't find that information. Please try being more specific."
            
        except Account.DoesNotExist:
            logger.error("Account not found for user")
            return "Account not found. Please contact support."
        except Exception as e:
            logger.error(f"Database access error: {str(e)}")
            return "Unable to fetch data at this time."
        
    def _format_transactions(self, transactions) -> str:
        """Format transaction data for response"""
        if not transactions.exists():
            return "No recent transactions found."
            
        serialized = TransactionsSerializer(transactions, many=True).data
        return "Recent transactions:\n" + "\n".join(
            f"{t['amount']:,.2f} NPR - {t['description']} ({t['date']})"
            for t in serialized
        )
    
    def _get_loan_details(self, account) -> str:
        """Format loan information for response with better error handling"""
        if not account.loanID:
            return "You don't have any active loans."
            
        try:
            loan = Loans.objects.get(pk=account.loanID.pk)
            data = LoansSerializer(loan).data
            
            # Ensure all values are properly formatted
            loan_amount = float(data.get('loanAmount', 0))
            outstanding = float(data.get('outstandingAmount', 0))
            rate = float(data.get('interestRate', 0))
            
            return (
                f"Loan Details:\n"
                f"Type: {data.get('type', 'N/A')}\n"
                f"Amount: NPR {loan_amount:,.2f}\n"
                f"Outstanding: NPR {outstanding:,.2f}\n"
                f"Interest Rate: {rate:.2f}%"
            )
        except Exception as e:
            logger.error(f"Loan details formatting error: {str(e)}")
            return "Could not format loan details."
    
    def _financial_calculator(self, query: str) -> str:
        """Handle financial calculations"""
        try:
            amounts = [float(x) for x in re.findall(r'\d+\.?\d*', query)]
            
            if "interest" in query.lower() and len(amounts) >= 2:
                return self._calculate_interest(amounts)
            elif "emi" in query.lower() and len(amounts) >= 3:
                return self._calculate_emi(amounts)
                
            return "Please provide all required values (amount, rate, duration)"
            
        except Exception as e:
            logger.error(f"Calculation error: {str(e)}")
            return "Unable to perform calculation"
    
    def _calculate_interest(self, amounts) -> str:
        principal, rate = amounts[0], amounts[1]
        years = amounts[2] if len(amounts) > 2 else 1
        interest = principal * (rate/100) * years
        return (
            f"Interest Calculation:\n"
            f"Principal: NPR {principal:,.2f}\n"
            f"Rate: {rate}%\n"
            f"Time: {years} year(s)\n"
            f"Interest: NPR {interest:,.2f}"
        )
    
    def _calculate_emi(self, amounts) -> str:
        principal, rate, months = amounts[0], amounts[1], amounts[2]
        monthly_rate = rate / 12 / 100
        emi = principal * monthly_rate * (1 + monthly_rate)**months / ((1 + monthly_rate)**months - 1)
        return (
            f"EMI Calculation:\n"
            f"Loan Amount: NPR {principal:,.2f}\n"
            f"Interest Rate: {rate}% p.a.\n"
            f"Tenure: {months} months\n"
            f"Monthly EMI: NPR {emi:,.2f}"
        )
    
    def _currency_converter(self, query: str) -> str:
        """Handle currency conversions"""
        try:
            amounts = [float(x) for x in re.findall(r'\d+\.?\d*', query)]
            currencies = re.findall(r'[A-Z]{3}', query.upper())
            
            if len(amounts) >= 1 and len(currencies) >= 2:
                amount = amounts[0]
                from_curr, to_curr = currencies[0], currencies[1]
                rate = self._get_exchange_rate(from_curr, to_curr)
                converted = amount * rate
                return (
                    f"Currency Conversion:\n"
                    f"Amount: {amount} {from_curr}\n"
                    f"Rate: 1 {from_curr} = {rate:.2f} {to_curr}\n"
                    f"Result: {converted:.2f} {to_curr}"
                )
            return "Please specify amount and currencies (e.g., 'convert 100 USD to EUR')"
        except Exception as e:
            logger.error(f"Conversion error: {str(e)}")
            return "Unable to perform conversion"
    
    def _get_exchange_rate(self, from_curr: str, to_curr: str) -> float:
        """Mock exchange rate service"""
        rates = {
            "USD_NPR": 133.50,
            "EUR_NPR": 145.25,
            "GBP_NPR": 170.80,
            "NPR_USD": 1/133.50,
            "NPR_EUR": 1/145.25,
            "NPR_GBP": 1/170.80
        }
        return rates.get(f"{from_curr}_{to_curr}", 1.0)
    
    def _generate_steps_response(self, query: str) -> str:
        """Generate procedural instructions"""
        prompt = f"""Provide clear, numbered steps for this banking request:
        {query}
        
        Instructions:"""
        return self.llm.invoke(prompt)
    
    def process_query(self, query: str) -> BankingResponse:
        """Process query with classification-based routing"""
        try:
            if not query.strip():
                return BankingResponse(
                    query="",
                    type=QueryType.STEPS,
                    response="Please enter a valid query",
                    confidence=0.1
                )
            
            # Classify query
            query_type = self.classifier.classify(query)
            logger.debug(f"Classified '{query}' as {query_type}")
            
            # Route based on type
            if query_type == QueryType.DIRECT:
                response = self._fetch_from_database(query)
            elif query_type == QueryType.STEPS:
                response = self._generate_steps_response(query)
            else:
                response = self.agent.invoke({"input": query})["output"]
            
            # Update chat history
            self._update_chat_history(query, response)
            
            return BankingResponse(
                query=query,
                type=query_type,
                response=response,
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            return BankingResponse(
                query=query,
                type=QueryType.STEPS,
                response="I couldn't process your request. Please try again.",
                confidence=0.1
            )
        
    def _update_chat_history(self, query: str, response: str):
        """Maintain conversation history"""
        self.message_history.add_messages([
            HumanMessage(content=query),
            AIMessage(content=response)
        ])
        # Trim history if needed
        if len(self.message_history.messages) > self.max_history:
            self.message_history.messages = self.message_history.messages[-self.max_history:]