from datetime import datetime
import re
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from langchain_community.llms import LlamaCpp
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, validator
from django.db.models import Q
from .models import Account, CurrencyExchange, LoanAccount, Loans, Transactions
from .serializers import AccountSerializer, LoansSerializer, TransactionsSerializer
from langchain.memory import ConversationBufferMemory, ChatMessageHistory

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
    context: Optional[dict] = None
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

class QueryClassifier:
    """Handles query classification with precise pattern matching"""
    def __init__(self, llm: LlamaCpp):
        self.llm = llm
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize pre-compiled regex patterns"""
        
        self.pattern_map = {
            QueryType.CALCULATIONS: [
            re.compile(r'calculat(e|ion)', re.IGNORECASE),
                re.compile(r'comput(e|ation)', re.IGNORECASE),
                re.compile(r'\bemi\b', re.IGNORECASE), # Moved EMI up
                # Move interest/installment patterns to top
                re.compile(r'\b(next|upcoming)\s+(month\'?s?)?\s*(interest|installment|payment)\b', re.IGNORECASE),
                re.compile(r'\b(how much|what is)\s+(my|the)\s+(next|upcoming)\s+interest\b', re.IGNORECASE),
                re.compile(r'\b(calculate|compute)\s+(next|upcoming)\s+interest\b', re.IGNORECASE),
                re.compile(r'installment interest', re.IGNORECASE),
                re.compile(r'next (payment|installment)', re.IGNORECASE),
                re.compile(r'how much interest', re.IGNORECASE),
            
                # Keep other calculation patterns
                re.compile(r'calculat(e|ion)', re.IGNORECASE),
                re.compile(r'comput(e|ation)', re.IGNORECASE),
                re.compile(r'\bemi\b', re.IGNORECASE),

                # --- NEW/Enhanced Currency/Rate Patterns ---
                re.compile(r'\b(what is\s+the\s+)?exchange\s+rate(s)?\b', re.IGNORECASE),  # Added for exchange rate queries
                re.compile(r'\b(convert|exchange|change)\b', re.IGNORECASE), # Explicit conversion actions
                re.compile(r'\b(foreign\s+exchange|forex|currency)\s+rate(s)?\b', re.IGNORECASE), # Asking for rates
                re.compile(r'\b(rate(s)?\s+(for|of|between))\b', re.IGNORECASE), # More rate phrasings
                re.compile(r'how\s+much\s+.*?\s+(is|in)\s+.*?', re.IGNORECASE), # How much X is Y / How much X in Y
                re.compile(r'\d+\s*(dollar|euro|pound|yen|rupee|usd|eur|gbp|jpy|inr|npr)s?\s+(to|in|into)', re.IGNORECASE), # Specific conversion format N CUR to/in...
                re.compile(r'(to|in|into)\s+\d+\s*(dollar|euro|pound|yen|rupee|usd|eur|gbp|jpy|inr|npr)s?', re.IGNORECASE), # Specific conversion format ...to/in N CUR
                re.compile(r'\b(usd|eur|gbp|jpy|inr|npr)\s+(to|in|into)\s+(usd|eur|gbp|jpy|inr|npr)\b', re.IGNORECASE), # CUR to CUR

                # Existing patterns (refined slightly)
                re.compile(r'\d+\s*%\s+of\s+\d+', re.IGNORECASE), # Percentage calculation
                re.compile(r'installment interest', re.IGNORECASE),
                re.compile(r'next (payment|installment)', re.IGNORECASE),
                re.compile(r'how much interest', re.IGNORECASE),
                re.compile(r'\d+\s*(month|year)s?\s+at\s+\d+%', re.IGNORECASE), # Loan term/rate format
                re.compile(r'next (interest|installment|payment)', re.IGNORECASE), # Duplicates removed
                re.compile(r'(upcoming|next month\'?s?) interest', re.IGNORECASE),
                re.compile(r'\b(what is|calculate|how much)\s+my\s+interest\s+for\s+next\s+month\b', re.IGNORECASE),
                re.compile(r'\binterest\s+for\s+next\s+month\b', re.IGNORECASE),
            ],
            
            QueryType.STEPS: [
                re.compile(r'how to', re.IGNORECASE),
                re.compile(r'steps to', re.IGNORECASE),
                re.compile(r'process (to|for)', re.IGNORECASE),
                re.compile(r'what (do|should) i do to', re.IGNORECASE),
                re.compile(r'way to', re.IGNORECASE),
                re.compile(r'about', re.IGNORECASE),
                re.compile(r'\binsurance\b', re.IGNORECASE), 
                
            ],
            QueryType.DIRECT: [
                re.compile(r'\b(my|check|view|show)\s+(balance|transactions?|loans?)\b', re.IGNORECASE),
                re.compile(r'\b(account\s+statement|loan\s+status)\b', re.IGNORECASE),
                re.compile(r'\btypes?\s+of\s+loans?\b', re.IGNORECASE),
                re.compile(r'\b(tell me more|details?)\s+about\b', re.IGNORECASE),
                re.compile(r'\binterest\s+rate(s)?\b', re.IGNORECASE),
                re.compile(r'\b(send|transfer)\s+money\b', re.IGNORECASE),
                re.compile(r'\b(criteria|requirements|eligibility|necessary|need|required)\s+(for|to)\s+(loan|education loan|personal loan)\b', re.IGNORECASE),
                re.compile(r'\bwhat (do|should) i need (for|to get)\s+(a|an)\s+loan\b', re.IGNORECASE),
                re.compile(r'\b(requirements?|criteria|eligibility|documents? needed|papers? required|what (do|does) i need)\b', re.IGNORECASE),
            ],
            
            "OFF_TOPIC": [
                re.compile(r'\b(trump|biden|politics|sports|weather|movie)\b', re.IGNORECASE),
                re.compile(r'^who (is|are)', re.IGNORECASE),
                re.compile(r'^what is', re.IGNORECASE),
                re.compile(r'^what is\s+(a|an|the)\s+(?!exchange|currency|forex|interest|rate|emi|loan)', re.IGNORECASE), # Avoid banking terms
            ]
        }

    def _is_banking_related(self, query: str) -> bool:
        """Check if query is banking-related"""
        banking_keywords = self._get_banking_keywords()
        query_lower = query.lower()
        
        # Explicitly allow currency conversion queries
        currency_indicators = {'convert', 'exchange', 'change', 'currency', 'forex', 'rate', 
                             'usd', 'npr', 'inr', 'eur', 'gbp', 'jpy', 'rupee', 'dollar', 'euro', 'pound', 'yen'}
        if any(indicator in query_lower for indicator in currency_indicators):
            logger.debug(f"Query recognized as banking-related due to currency indicators: {query}")
            return True
        
        # Check for off-topic patterns
        if any(pattern.search(query) for pattern in self.pattern_map["OFF_TOPIC"]):
            if not any(keyword in query_lower for keyword in banking_keywords):
                logger.debug(f"Query matched OFF_TOPIC pattern and lacks banking keywords: {query}")
                return False
        
        # General banking keyword check
        is_related = any(keyword in query_lower for keyword in banking_keywords)
        logger.debug(f"Banking-related check for '{query}': {is_related}")
        return is_related
    
    def _get_banking_keywords(self) -> set:
        """Centralized list of banking keywords"""
        return {
            'balance', 'account', 'loan', 'transaction', 'interest', 'emi',
            'transfer', 'money', 'currency', 'bank', 'deposit', 'payment',
            'withdrawal', 'foreign exchange', 'forex', 'rate', 'convert',
            'exchange', 'usd', 'npr', 'inr', 'eur', 'gbp', 'jpy', 'statement',
            'apply', 'eligibility', 'credit', 'debit', 'fund'
        }
    def _pattern_match(self, query: str) -> Optional[QueryType]:
        query_lower = query.lower()
        # Prioritize STEPS for "how to" queries
        if query_lower.startswith('how to'):
            for pattern in self.pattern_map[QueryType.STEPS]:
                if pattern.search(query):
                    logger.debug(f"Matched STEPS pattern {pattern.pattern} for query: {query}")
                    return QueryType.STEPS
        
        # Check CALCULATIONS first for currency conversions
        for pattern in self.pattern_map[QueryType.CALCULATIONS]:
            if pattern.search(query):
                logger.debug(f"Matched CALCULATIONS pattern {pattern.pattern} for query: {query}")
                return QueryType.CALCULATIONS
        
        # Then check DIRECT
        for pattern in self.pattern_map[QueryType.DIRECT]:
            if pattern.search(query):
                logger.debug(f"Matched DIRECT pattern {pattern.pattern} for query: {query}")
                return QueryType.DIRECT
        
        # Finally check STEPS
        for pattern in self.pattern_map[QueryType.STEPS]:
            if pattern.search(query):
                logger.debug(f"Matched STEPS pattern {pattern.pattern} for query: {query}")
                return QueryType.STEPS
        
        return None

    def classify(self, query: str) -> QueryType:
        """Classify query with robust error handling and improved currency detection"""
        
        
        if not self._is_banking_related(query):
            raise ValueError("This query is not related to banking.")
        
        try:
            query_lower = query.lower()
            # First check for specific interest calculation patterns
            interest_phrases = [
                'next interest',
                'next month interest',
                'upcoming interest',
                'installment interest',
                'next payment interest',
                'how much interest will i pay'
            ]
            
            if any(phrase in query_lower for phrase in interest_phrases):
                logger.debug(f"Matched interest calculation pattern for query: {query}")
                return QueryType.CALCULATIONS
            
            calc_keywords = ['calculate', 'computation', 'convert', 'exchange', 'change', 'emi', 'rate', 'forex', 'foreign exchange', 'how much']
            currency_indicators = ['rupee','rupees', 'npr', 'inr', 'npr', 'dollar', 'euro', 'pound', 'yen', 'usd', 'eur', 'gbp', 'jpy', '%', 'interest']
            contains_number = re.search(r'\d', query)
            
            is_likely_calculation = False
            if any(term in query_lower for term in calc_keywords):
                # If keywords like convert/exchange/rate/emi are present, it's highly likely a calculation
                if any(term in query_lower for term in ['convert', 'exchange', 'change', 'rate', 'emi', 'forex', 'calculate', 'computation']):
                     is_likely_calculation = True
                     logger.debug(f"Strong CALCULATION keyword detected: {query}")
                # If "how much" is present with numbers or currency indicators, lean towards calculation
                elif 'how much' in query_lower and (contains_number or any(ind in query_lower for ind in currency_indicators)):
                     is_likely_calculation = True
                     logger.debug(f"'how much' + indicators points to CALCULATION: {query}")

            # If keywords + indicators are present, also likely calculation
            elif contains_number and any(ind in query_lower for ind in currency_indicators):
                 is_likely_calculation = True
                 logger.debug(f"Numbers + currency indicators point to CALCULATION: {query}")


            if is_likely_calculation:
                 # Try pattern matching for CALCULATION first for confirmation/specificity
                 for pattern in self.pattern_map[QueryType.CALCULATIONS]:
                    if pattern.search(query):
                        logger.debug(f"Confirmed CALCULATIONS via pattern {pattern.pattern} after keyword check: {query}")
                        return QueryType.CALCULATIONS
                 # If keywords strongly suggested calculation, but no specific pattern matched, classify as CALCULATION anyway
                 logger.debug(f"Classifying as CALCULATIONS based on keyword/indicator logic, despite no specific pattern match: {query}")
                 return QueryType.CALCULATIONS
            # --- End Enhanced Check ---


            # If not strongly identified as calculation, proceed with normal pattern matching order
            query_type = self._pattern_match(query)
            if query_type:
                logger.debug(f"Pattern match result (after enhanced check): {query_type}")
                return query_type

            # Final fallback if no patterns matched
            # If it contains numbers, maybe it's a calculation missed? Or direct (e.g., account number)? Defaulting to DIRECT is safer.
            logger.debug(f"No specific patterns matched, falling back to DIRECT for query: {query}")
            return QueryType.DIRECT

        except Exception as e:
            logger.error(f"Classification error for query '{query}': {str(e)}", exc_info=True)
            return QueryType.STEPS  # Safe fallback in case of unexpected errors

class BankingAssistant:
    """Main banking assistant class with context-aware capabilities"""
    
    # Add this at the class level in BankingAssistant
    CURRENCY_MAPPING = {
    # Full names and common abbreviations
    'us dollar': 'USD', 'us dollars': 'USD', 'dollar': 'USD', 'dollars': 'USD', '$': 'USD',
    'euro': 'EUR', 'euros': 'EUR', '€': 'EUR',
    'pound': 'GBP', 'pounds': 'GBP', 'sterling': 'GBP', '£': 'GBP',
    'swiss franc': 'CHF', 'franc': 'CHF',
    'australian dollar': 'AUD',
    'canadian dollar': 'CAD',
    'singapore dollar': 'SGD',
    'japanese yen': 'JPY', 'yen': 'JPY', '¥': 'JPY',
    'chinese yuan': 'CNY', 'yuan': 'CNY', 'renminbi': 'CNY',
    'saudi riyal': 'SAR', 'riyal': 'SAR',
    'qatari riyal': 'QAR',
    'thai baht': 'THB', 'baht': 'THB', '฿': 'THB',
    'uae dirham': 'AED', 'dirham': 'AED',
    'malaysian ringgit': 'MYR', 'ringgit': 'MYR', 'rm': 'MYR',
    'korean won': 'KRW', 'won': 'KRW', '₩': 'KRW',
    'swedish krona': 'SEK', 'krona': 'SEK',
    'danish krone': 'DKK', 'krone': 'DKK',
    'hong kong dollar': 'HKD',
    'kuwaiti dinar': 'KWD', 'dinar': 'KWD',
    'bahraini dinar': 'BHD',
    'omani rial': 'OMR', 'rial': 'OMR',
    
    # Indian and Nepali currencies (from previous)
    'indian rupee': 'INR', 'indian rupees': 'INR', 'inr': 'INR', '₹': 'INR',
    'nepali rupee': 'NPR', 'nepali rupees': 'NPR', 'npr': 'NPR', 'रू': 'NPR'
}
    def __init__(self, llm: LlamaCpp, user_id: int = None, max_history: int = 10):
        self.llm = llm
        self.user_id = user_id
        self.max_history = max_history
        self.message_history = ChatMessageHistory()
        self.context = {
            'last_topics': [],
            'current_focus': None,
            'pending_actions': [],
            'last_action': None,
            'previous_loan_type': None
        }
        
        self.classifier = QueryClassifier(llm)
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent()
    
    def _initialize_tools(self) -> list:
        """Initialize the tools available to the agent"""
        return [
            Tool(
                name="BankingDatabase",
                func=self._handle_database_query,
                description=(
                    "Useful for all banking information including:"
                    "- Account balances and transactions "
                    "- Loan details and status "
                    "- Available loan products and their terms "
                    "- Interest rates and eligibility criteria "
                )
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
        """Initialize the LangChain agent"""
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
    
    def _handle_database_query(self, query: str) -> str:
        """Centralized database query handler"""
        query_lower = query.lower()
        
        # Check for authentication requirement
        if not self.user_id and any(word in query_lower for word in ['my', 'mine']):
            return "Please log in to access your account information."
        
        # Route to appropriate handler
        if "loan" in query_lower:
            return self._handle_loan_query(query)
        elif "transaction" in query_lower:
            return self._handle_transaction_query(query)
        elif "balance" in query_lower:
            return self._handle_balance_query()
        elif "account" in query_lower:
            return self._handle_account_query()
            
        return "I couldn't find that information. Please try being more specific."

    def _handle_loan_query(self, query: str) -> str:
        """Comprehensive loan query handler that replaces multiple functions"""
        query_lower = query.lower()
        
        # 1. Loan Products Listing
        if any(phrase in query_lower for phrase in ['types of loan', 'loan products', 'what loans']):
            return self._get_loan_products_list()
        
        # 2. Loan Criteria/Requirements
        elif any(phrase in query_lower for phrase in ['criteria', 'requirements', 'eligibility', 'necessary', 'need', 'required']):
            return self._get_loan_criteria(query)
        
        # 2. Personal Loan Status
        elif any(phrase in query_lower for phrase in ['my loan', 'loan status']):
            return self._get_personal_loan_status()
        
        # Interest calculation only for active loans
        elif any(phrase in query_lower for phrase in [
            'next interest', 
            'installment interest',
            'payment interest',
            'next month interest',
            'interest for next month',
            'next month interest'
        ]):
            return self._calculate_installment_interest(query)
        
        # 3. Specific Loan Details
        elif any(word in query_lower for word in ['home loan', 'personal loan', 'business loan']):
            return self._get_specific_loan_info(query_lower)
        
        # 4. Interest Rate Queries
        elif "interest rate" in query_lower or "interest" in query_lower:
            return self._handle_interest_rate_question(query_lower)
        
        # 5. Calculation Requests
        elif any(term in query_lower for term in ['calculate', 'emi', 'monthly payment']):
            return self._financial_calculator(query)
        
        return self._get_loan_help_message()
    
    def _get_loan_criteria(self, query: str) -> str:
        """Handle loan eligibility/criteria questions"""
        query_lower = query.lower()
        
        # Identify loan type from query
        loan_type = None
        for product in Loans.objects.filter(is_active=True):
            if product.loanType.lower() in query_lower:
                loan_type = product
                break
        
        if not loan_type:
            # If no specific loan mentioned, show general criteria
            return (
                "General loan requirements:\n"
                "1. Valid citizenship certificate\n"
                "2. Minimum age: 18 years\n"
                "3. Regular income source\n"
                "4. Good credit history\n\n"
                "Please specify a loan type for specific criteria (e.g., 'education loan requirements')."
            )
        
        # Loan-specific criteria
        response = f"Requirements for {loan_type.loanType}:\n"
        
        if loan_type.loanType.lower() == "education loan":
            response += (
                "1. Admission letter from recognized institution\n"
                "2. Fee structure from the institution\n"
                "3. Parent/guardian as co-signer\n"
                "4. Academic transcripts\n"
                f"5. Minimum amount: NPR {loan_type.minAmount:,.2f}\n"
                f"Interest rate: {loan_type.interestRate}%\n\n"
                "Would you like to apply for this loan?"
            )
        elif loan_type.loanType.lower() == "personal loan":
            response += (
                "1. 3 months salary slips\n"
                "2. Employment verification\n"
                "3. Bank statements (6 months)\n"
                f"4. Minimum amount: NPR {loan_type.minAmount:,.2f}\n"
                f"Interest rate: {loan_type.interestRate}%\n\n"
                "Apply at any branch with these documents."
            )
        else:
            response += (
                f"1. Minimum amount: NPR {loan_type.minAmount:,.2f}\n"
            f"2. Maximum amount: NPR {loan_type.maxAmount:,.2f}\n"
            f"3. Interest rate: {loan_type.interestRate}%\n"
            f"4. Minimum term: {loan_type.minTerm} months\n"
            f"5. Maximum term: {loan_type.maxTerm} months\n"
            "6. Valid citizenship document\n\n"
            "Visit our website or branch for complete details."
            )
        
        return response

    def _suggest_loan_types(self, loans) -> str:
        """Suggest available loan types when none is specified"""
        loan_names = [loan.loanType for loan in loans]
        return (
            f"Please specify which loan you're interested in. "
            f"We offer: {', '.join(loan_names)}. "
            f"For example: 'What's the interest rate for {loan_names[0]}?'"
        )
    
    def _handle_transaction_query(self, query: str) -> str:
        """Handle transaction history queries with dynamic limit"""
        try:
            # 1. Authentication check
            if not self.user_id:
                return "Please log in to view your transaction history."
    
            # 2. Parse requested transaction count (default to 5)
            count = 5
            if "last transaction" in query.lower():
                count = 1
            else:
                numbers = re.findall(r'\d+', query)
                if numbers:
                    count = min(int(numbers[0]), 20)  # Max 20 transactions
    
            # 3. Get transactions
            account = Account.objects.get(user__id=self.user_id)
            transactions = Transactions.objects.filter(
                accountID=account
            ).order_by('-date')[:count]
            
            if not transactions.exists():
                return "No transactions found."
                
            # 4. Return serialized data
            serializer = TransactionsSerializer(transactions, many=True)
            return str(serializer.data)
    
        except Account.DoesNotExist:
            return "Account not found. Please contact customer support."
        except Exception as e:
            logger.error(f"Transaction error: {str(e)}")
            return "Unable to retrieve transactions. Please try again later."
    
    def _handle_balance_query(self) -> str:
        """Handle balance queries"""
        if not self.user_id:
            return "Please log in to view your account balance."
            
        try:
            account = Account.objects.get(user__id=self.user_id)
            balance = float(AccountSerializer(account).data['balance'])
            return f"Your current balance is NPR {balance:,.2f}"
        except Exception as e:
            logger.error(f"Balance query error: {str(e)}")
            return "Unable to retrieve balance information at this time."
    
    def _handle_account_query(self) -> str:
        """Handle general account queries"""
        if not self.user_id:
            return "Please log in to access your account information."
            
        try:
            account = Account.objects.get(user__id=self.user_id)
            return (
                f"Your Account Summary:\n"
                f"- Account Number: {account.accountNumber}\n"
                f"- Balance: NPR {account.balance:,.2f}\n"
                f"- Status: {account.status}"
            )
        except Exception as e:
            logger.error(f"Account query error: {str(e)}")
            return "Unable to retrieve account information at this time."
    
    def _format_transactions(self, transactions) -> str:
        if not transactions.exists():
            return "No recent transactions found."
            
        return "Recent transactions:\n" + "\n".join(
            f"{t.amount:,.2f} NPR - {t.description} ({t.date})"  # Using direct model fields
            for t in transactions
        )
    
    def _financial_calculator(self, query: str) -> str:
        """Handle financial calculations using database data for loan interest queries"""
        try:
            query_lower = query.lower()
            logger.debug(f"Financial calculator input query: {query}, matched terms: {query_lower}")
            
            # Handle installment interest queries using database data
            if any(term in query_lower for term in [
                'next interest',
                'next installment',
                'next payment',
                'upcoming interest',
                'how much interest will i pay',
                'installment interest',
                'interest for next month',  # Added for queries like "calculate my interest for next month"
                'next month interest',
                'interest next month'
            ]):
                return self._calculate_installment_interest(query)
            
            # Extract numbers for explicit calculations
            amounts = [float(x) for x in re.findall(r'\d+\.?\d*', query)]
            
            # Try to get rate if specified with % sign
            rate_match = re.search(r'(\d+\.?\d*)%', query)
            if rate_match:
                amounts.insert(1, float(rate_match.group(1)))  # Insert rate at position 1
                
            # Handle different calculation types
            if "emi" in query_lower:
                if len(amounts) >= 3:
                    return self._calculate_emi(amounts)
                return (
                    "Please provide all required values for EMI calculation:\n"
                    "1. Loan amount (e.g., 100000)\n"
                    "2. Interest rate (e.g., 10.5%)\n"
                    "3. Duration in months (e.g., 24)"
                )
            elif "interest" in query_lower:
                if len(amounts) >= 2:
                    return self._calculate_interest(amounts)
                return (
                    "Please provide for manual interest calculation:\n"
                    "1. Principal amount (e.g., 50000)\n"
                    "2. Interest rate (e.g., 8%)\n"
                    "Or try 'calculate my next month interest' for your loan details."
                )
                
            return (
                "I can help with:\n"
                "- EMI calculations (say 'calculate EMI for 100000 at 10% for 24 months')\n"
                "- Interest calculations (say 'calculate interest on 50000 at 8%')\n"
                "- Loan interest (say 'calculate my next month interest')"
            )
            
        except Exception as e:
            logger.error(f"Calculation error: {str(e)}")
            return "Unable to perform calculation. Please provide clear numbers or try 'calculate my next month interest'."
        
    def _calculate_interest(self, amounts) -> str:
        """Calculate simple interest"""
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
        """Calculate EMI for loans"""
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
    

        """Calculate EMI for loans"""
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
    
    def _calculate_installment_interest(self, query: str) -> str:
        """Calculate interest for next installment based on user's active loan"""
        try:
            if not self.user_id:
                return "Please log in to view your loan details."
            
            # Get only ACTIVE loans
            active_loans = LoanAccount.objects.filter(
            account__user__id=self.user_id,
            status='active'  # Only active loans
        )
        
            if not active_loans.exists():
            # Helpful message that distinguishes no loans vs no active loans
                all_loans = LoanAccount.objects.filter(account__user__id=self.user_id)
                if all_loans.exists():
                    return "You have no active loans currently. Your loans are pending approval."
                return "You don't have any loans."
        
            loan = active_loans.first()
            monthly_interest = (loan.outstanding * loan.interest_rate) / (12 * 100)

            
            # Calculate days until next payment (for more accurate daily interest if needed)
            from datetime import date
            days_until_payment = (loan.next_payment_date - date.today()).days
            
            return (
                f"Next Installment Interest Calculation for your {loan.product.loanType}:\n"
                f"- Outstanding Principal: NPR {loan.outstanding:,.2f}\n"
                f"- Annual Interest Rate: {loan.interest_rate}%\n"
                f"- Monthly Interest: NPR {monthly_interest:,.2f}\n"
                f"- Next Payment Due: {loan.next_payment_date} (in {days_until_payment} days)\n\n"
                f"Note: Your actual payment may include both principal and interest components."
            )
            
        except Exception as e:
            logger.error(f"Installment interest calculation error: {str(e)}")
            return "Unable to calculate your installment interest. Please try again later."

    def _get_loan_products_list(self) -> str:
        """List all available loan products"""
        products = Loans.objects.filter(is_active=True)
        if not products.exists():
            return "No loan products currently available."
        
        response = "Available loan products:\n"
        for product in products:
            response += (
                f"- {product.loanType}: "
                f"NPR {product.minAmount or 0:,.2f}-{product.maxAmount or 0:,.2f} "
                f"at {product.interestRate}% for "
                f"{product.minTerm or 0}-{product.maxTerm or 0} months\n"
            )
        return response + "\nAsk about a specific loan for more details."

    def _get_personal_loan_status(self) -> str:
        """Get user's personal loan details"""
        if not self.user_id:
            return "Please log in to view your loan details."
        
        try:
            loans = LoanAccount.objects.filter(account__user__id=self.user_id)
            if not loans.exists():
                return "You don't have any active loans."
            
            response = "Your loan details:\n"
            for loan in loans:
                status_display = {
                'pending': 'Pending Approval',
                'active': 'Active',
                'paid': 'Paid Off',
                'defaulted': 'Defaulted',
                'rejected': 'Rejected'
            }.get(loan.status, loan.status)
                response += (
                    f"- {loan.product.loanType}:\n"
                    f"  Amount: NPR {loan.amount:,.2f}\n"
                    f"  Outstanding: NPR {loan.outstanding:,.2f}\n"
                    f"  Interest: {loan.interest_rate}%\n"
                    f"  Next Payment: {loan.next_payment_date}\n"
                )
            return response
        
        except Exception as e:
            logger.error(f"Loan status error: {str(e)}")
            return "Unable to retrieve your loan information."
    
    def _get_specific_loan_info(self, query: str) -> str:
        """Get details for a specific loan type"""
        for product in Loans.objects.filter(is_active=True):
            if product.loanType.lower() in query:
                return (
                    f"{product.loanType} Details:\n"
                    f"- Rate: {product.interestRate}%\n"
                    f"- Amount: NPR {product.minAmount:,.2f} to {product.maxAmount:,.2f}\n"
                    f"- Term: {product.minTerm} to {product.maxTerm} months\n"
                    f"Apply with just your citizenship certificate!"
                )
        return "Could not find that loan product."
    
    def _handle_interest_rate_question(self, query: str) -> str:
        """Handle interest rate queries"""
        if "for" in query:  # Specific loan type
            for product in Loans.objects.filter(is_active=True):
                if product.loanType.lower() in query:
                    return f"Our {product.loanType} has an interest rate of {product.interestRate}%."
        
        # General interest rates
        rates = [f"{p.loanType}: {p.interestRate}%" for p in Loans.objects.filter(is_active=True)]
        return "Current interest rates:\n" + "\n".join(rates)
    
    def _get_loan_help_message(self) -> str:
        """Default loan help message"""
        return (
            "I can help with:\n"
            "- Types of loans we offer\n"
            "- Your loan status\n"
            "- Interest rates\n"
            "- EMI calculations\n"
            "Try asking about a specific loan type like 'home loan'"
        )
        
    def _currency_converter(self, query: str) -> str:
        """Convert currency based on user query, with fallback for API failures"""
        try:
            query_lower = query.replace(',', '').lower()

            # Extract amount
            amount_match = re.search(r'(\d+\.?\d*)', query_lower)
            if not amount_match:
                logger.debug(f"No amount found in query: {query}")
                return "Please specify an amount to convert (e.g., 'convert 100 INR to NPR')"

            amount = float(amount_match.group(1))
            logger.debug(f"Parsed amount: {amount}")

            # Initialize currencies
            from_currency = None
            to_currency = None

            # Currency mapping
            CURRENCY_MAPPING = {
                'dollar': 'USD', 'dollars': 'USD', 'usd': 'USD',
                'rupee': 'NPR', 'rupees': 'NPR', 'nepali rupee': 'NPR', 'npr': 'NPR',
                'indian rupee': 'INR', 'inr': 'INR',
                'euro': 'EUR', 'euros': 'EUR', 'eur': 'EUR',
                'pound': 'GBP', 'pounds': 'GBP', 'gbp': 'GBP',
                'yen': 'JPY', 'jpy': 'JPY'
            }

            # Extract currencies
            words = query_lower.split()
            for word in words:
                code = CURRENCY_MAPPING.get(word)
                if code:
                    if not from_currency:
                        from_currency = code
                    elif not to_currency and code != from_currency:
                        to_currency = code

            # Additional check for currency codes
            currency_codes = ['usd', 'npr', 'inr', 'eur', 'gbp', 'jpy']
            for word in words:
                if word in currency_codes:
                    code = word.upper()
                    if not from_currency:
                        from_currency = code
                    elif not to_currency and code != from_currency:
                        to_currency = code

            if not from_currency or not to_currency:
                logger.debug(f"Failed to parse currencies: from={from_currency}, to={to_currency}")
                return "Please specify both source and target currencies (e.g., 'convert 100 INR to NPR')"

            logger.debug(f"Parsed currencies: {from_currency} to {to_currency}")

            # Get current date
            from datetime import date
            today = date.today().isoformat()

            # Perform conversion
            try:
                conversion = CurrencyExchange.convert_currency(
                    amount=amount,
                    date=today,
                    from_currency=from_currency,
                    to_currency=to_currency
                )

                if not conversion.get("success"):
                    logger.warning(f"Conversion failed: {conversion.get('error')}")
                    raise ValueError(f"API returned error: {conversion.get('error')}")

                result = conversion["converted_amount"]
                rate = conversion["exchange_rate"]
                source = "NRB Forex API (Today's rate)"

            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"Currency conversion failed for {from_currency}-{to_currency}: {str(e)}")
                # Fallback rates
                FALLBACK_RATES = {
                    ('INR', 'NPR'): 1.6,  # From your context
                    ('USD', 'NPR'): 133.5,
                    ('EUR', 'NPR'): 142.0,
                    ('NPR', 'INR'): 1/1.6,
                    ('NPR', 'USD'): 1/133.5,
                    ('NPR', 'EUR'): 1/142.0
                }
                rate = FALLBACK_RATES.get((from_currency, to_currency))
                if rate:
                    result = amount * rate
                    source = "Fallback Rate (Indicative)"
                else:
                    logger.error(f"No fallback rate available for {from_currency}-{to_currency}")
                    return f"Sorry, I couldn’t convert {from_currency} to {to_currency}. Please check the currencies and try again."

            return (
                f"Currency Conversion:\n"
                f"Amount: {amount:,.2f} {from_currency}\n"
                f"Rate: 1 {from_currency} = {rate:,.4f} {to_currency}\n"
                f"Result: {result:,.2f} {to_currency}\n"
                f"Source: {source}"
            )

        except Exception as e:
            logger.error(f"Unexpected currency conversion error: {str(e)}")
            return f"Sorry, I couldn’t convert {amount:,.2f} {from_currency or 'unknown'} to {to_currency or 'unknown'}. Please try again."
    
    def process_query(self, query: str) -> BankingResponse:
        """Main entry point with enhanced currency conversion handling"""
        try:
            query_lower = query.lower()
            query_type = self.classifier.classify(query)
            logger.debug(f"Classified '{query}' as {query_type}")
            
            # Initialize response to ensure it’s always defined
            response = ""
            source = "Agent/LLM"  # Default source
    
            # Force CALCULATIONS for interest queries
            interest_phrases = [
                'next interest',
                'next month interest',
                'upcoming interest',
                'installment interest',
                'next payment interest',
                'how much interest will i pay',
                'interest for next month'
            ]
            if any(phrase in query_lower for phrase in interest_phrases):
                query_type = QueryType.CALCULATIONS
                logger.debug(f"Overriding to CALCULATIONS for interest query: {query}")
            
            # Force CALCULATIONS for currency and exchange rate queries
            if any(term in query_lower for term in ['convert', 'exchange', 'change', 'rate', 'forex', 'foreign exchange']):
                query_type = QueryType.CALCULATIONS
                logger.debug(f"Overriding to CALCULATIONS for currency/rate query: {query}")
    
            # Handle based on query type
            if query_type == QueryType.CALCULATIONS:
                if any(term in query_lower for term in ['convert', 'exchange', 'change', 'rate', 'forex', 'foreign exchange']):
                    if 'rate' in query_lower and not any(indicator in query_lower for indicator in ['rupee', 'inr', 'npr', 'dollar', 'usd', 'euro', 'eur', 'pound', 'gbp']):
                        # Handle general exchange rate queries
                        response = self._currency_converter("1 USD to NPR")
                        source = "CurrencyConverter Tool (Default USD-NPR)"
                    else:
                        response = self._currency_converter(query)
                        source = "CurrencyConverter Tool"
                else:
                    response = self._financial_calculator(query)
                    source = "FinancialCalculator Tool"
                    if "interest" in query_lower and "month" in query_lower and "Unable to perform" in response:
                        response = self._calculate_installment_interest(query)
                        source = "FinancialCalculator Tool (Interest Fallback)"
            elif query_type == QueryType.DIRECT:
                if 'insurance' in query_lower:
                    response = "We currently don’t offer insurance services. How else can I assist with your banking needs?"
                    source = "Custom Response"
                else:
                    response = self._handle_database_query(query)
                    logger.debug("Handling DIRECT query in process_query")
            elif query_type == QueryType.STEPS:
                response = self._generate_steps_response(query)
            else:
                response = self.agent.invoke({"input": query})["output"]
    
            self._update_chat_history(query, response)
            return BankingResponse(
                query=query,
                type=query_type,
                response=response,
                confidence=0.9,
                context=self.context.copy()
            )
            
        except ValueError as ve:
            if any(term in query.lower() for term in ['convert', 'exchange', 'change', 'inr', 'npr', 'usd', 'eur', 'gbp']):
                response = "Please specify a currency conversion, e.g., 'convert 100 USD to NPR'."
            elif 'insurance' in query.lower():
                response = "We currently don’t offer insurance services. Try asking about loans, accounts, or currency exchange."
            else:
                response = "Please specify your banking query, e.g., 'check my balance' or 'convert 100 USD to NPR'."
            logger.debug(f"ValueError caught for query '{query}': {str(ve)}")
            self._update_chat_history(query, response)
            return BankingResponse(
                query=query,
                type=QueryType.STEPS,
                response=response,
                confidence=0.9
            )
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            response = "I couldn't process your request. Please try again with a specific banking query."
            self._update_chat_history(query, response)
            return BankingResponse(
                query=query,
                type=QueryType.STEPS,
                response=response,
                confidence=0.1
            )

    def _generate_steps_response(self, query: str) -> str:
        """Generate procedural instructions"""
        prompt = f"""Provide clear, numbered steps for this banking request:
        {query}
        
        Instructions:"""
        return self.llm.invoke(prompt)
    
    def _update_chat_history(self, query: str, response: str):
        """Maintain conversation history and update context"""
        self.message_history.add_messages([
            HumanMessage(content=query),
            AIMessage(content=response)
        ])
        
        # Trim history if needed
        if len(self.message_history.messages) > self.max_history:
            self.message_history.messages = self.message_history.messages[-self.max_history:]
            
        # Update context based on conversation
        self._update_context(query, response)
    
    def _update_context(self, query: str, response: str):
        """Update conversation context"""
        query_lower = query.lower()
        
        # Track last mentioned loan type
        if "loan" in query_lower:
            for loan in Loans.objects.all():
                if loan.loanType.lower() in query_lower:
                    self.context['current_focus'] = loan.loanType
                    break
        
        # Track specific actions
        if 'available loan types:' in response:
            self.context['last_action'] = 'listed_loan_types'
        elif 'interest rate' in response.lower():
            self.context['last_action'] = 'provided_interest_rate'