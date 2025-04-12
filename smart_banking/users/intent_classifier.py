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
from .models import Account, Loans, Transactions
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
            QueryType.STEPS: [
                re.compile(r'how to', re.IGNORECASE),
                re.compile(r'steps to', re.IGNORECASE),
                re.compile(r'process (to|for)', re.IGNORECASE),
                re.compile(r'what (do|should) i do to', re.IGNORECASE),
                re.compile(r'way to', re.IGNORECASE),
            ],
            QueryType.DIRECT: [
                re.compile(r'\b(my|check|view|show)\s+(balance|transactions?|loans?)\b', re.IGNORECASE),
                re.compile(r'\b(account\s+statement|loan\s+status)\b', re.IGNORECASE),
                re.compile(r'\btypes?\s+of\s+loans?\b', re.IGNORECASE),
                re.compile(r'\b(tell me more|details?)\s+about\b', re.IGNORECASE),
                re.compile(r'\binterest\s+rate(s)?\b', re.IGNORECASE),
                re.compile(r'\b(send|transfer)\s+money\b', re.IGNORECASE),
            ],
            QueryType.CALCULATIONS: [
                re.compile(r'calculat(e|ion)', re.IGNORECASE),
                re.compile(r'comput(e|ation)', re.IGNORECASE),
                re.compile(r'convert', re.IGNORECASE),
                re.compile(r'\d+\s*%\s+of\s+\d+', re.IGNORECASE),
                re.compile(r'emi', re.IGNORECASE),
            ],
            "OFF_TOPIC": [
                re.compile(r'\b(trump|biden|politics|sports|weather|movie)\b', re.IGNORECASE),
                re.compile(r'^who (is|are)', re.IGNORECASE),
                re.compile(r'^what is', re.IGNORECASE)
            ]
        }

    def _is_banking_related(self, query: str) -> bool:
        """Check if query is banking-related"""
        banking_keywords = {
            'balance', 'account', 'loan', 'transaction', 'interest', 'emi', 
            'transfer', 'money', 'currency', 'bank', 'deposit', 'payment',
            'withdrawal', 'foreign exchange', 'currency'
        }
        # Check for off-topic patterns first
        if any(pattern.search(query) for pattern in self.pattern_map["OFF_TOPIC"]):
            return False
        return any(keyword in query.lower() for keyword in banking_keywords)

    def _pattern_match(self, query: str) -> Optional[QueryType]:
        """Match query against predefined patterns"""
        for query_type, patterns in self.pattern_map.items():
            if query_type == "OFF_TOPIC":
                continue
            if any(pattern.search(query) for pattern in patterns):
                return query_type
        return None

    def classify(self, query: str) -> QueryType:
        """Classify query with robust error handling"""
        if not self._is_banking_related(query):
            raise ValueError("This query is not related to banking.")
        
        try:
            query_type = self._pattern_match(query)
            if query_type:
                return query_type
            return QueryType.DIRECT  # Default fallback
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return QueryType.STEPS  # Default fallback

class BankingAssistant:
    """Main banking assistant class with context-aware capabilities"""
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
            return self._handle_transaction_query()
        elif "balance" in query_lower:
            return self._handle_balance_query()
        elif "account" in query_lower:
            return self._handle_account_query()
            
        return "I couldn't find that information. Please try being more specific."
    
    def _handle_loan_query(self, query: str) -> str:
        """Handle all loan-related queries with context awareness"""
        query_lower = query.lower()
        
        # Check for personal loan details
        if any(phrase in query_lower for phrase in ["my loan", "loan status", "my loans"]):
            return self._get_user_loan_details()
            
        # Check for general loan information
        if any(phrase in query_lower for phrase in ['types of loan', 'loan types', 'available loans']):
            return self._get_all_loan_types()
            
        # Check for interest rate queries
        if "interest rate" in query_lower or "interest" in query_lower:
            return self._handle_interest_rate_query(query)
            
        # Specific loan details
        return self._get_specific_loan_details(query)
    
    def _handle_interest_rate_query(self, query: str) -> str:
        """Specialized handler for interest rate queries"""
        try:
            loans = Loans.objects.all()
            
            # Check if we have context about a specific loan
            if self.context.get('previous_loan_type'):
                for loan in loans:
                    if loan.loanType.lower() == self.context['previous_loan_type'].lower():
                        return (
                            f"The interest rate for {loan.loanType} is {loan.interestRate}%. "
                            f"Would you like to know more about {loan.loanType}?"
                        )
            
            # If asking about specific loan type
            for loan in loans:
                if loan.loanType.lower() in query.lower():
                    self.context['previous_loan_type'] = loan.loanType
                    return (
                        f"The interest rate for {loan.loanType} is {loan.interestRate}%. "
                        f"Would you like to know more about {loan.loanType}?"
                    )
            
            # If general interest rate question
            if "interest rate" in query.lower() or "interest" in query.lower():
                response = "Here are interest rates for our loan products:\n"
                response += "\n".join(
                    f"- {loan.loanType}: {loan.interestRate}%"
                    for loan in loans
                )
                response += "\n\nWhich loan product are you interested in?"
                self.context['last_action'] = 'listed_interest_rates'
                return response
                
            return "Could not find interest rate information. Please specify a loan type."
        except Exception as e:
            logger.error(f"Interest rate query error: {str(e)}")
            return "Unable to retrieve interest rates at this time."
    
    def _get_all_loan_types(self) -> str:
        """Retrieve all available loan types with context tracking"""
        try:
            loans = Loans.objects.all()
            if not loans.exists():
                return "No loan products currently available."
            
            loan_list = "\n".join(
                f"- {loan.loanType} (Interest: {loan.interestRate}%)" 
                for loan in loans
            )
            
            self.context['last_action'] = 'listed_loan_types'
            return (
                f"Available loan types:\n{loan_list}\n\n"
                f"You can ask about specific loans for more details, for example:\n"
                f"'What's the interest rate for home loans?'\n"
                f"'Tell me more about personal loans'"
            )
        except Exception as e:
            logger.error(f"Error getting loan types: {str(e)}")
            return "Unable to retrieve loan products at this time."
    
    def _get_specific_loan_details(self, query: str) -> str:
        """Get details for a specific loan type with context awareness"""
        try:
            loans = Loans.objects.all()
            target_loan = None
            
            # First try to match loan type from query
            for loan in loans:
                if loan.loanType.lower() in query.lower():
                    target_loan = loan
                    break
            
            # Check context if no direct match (follow-up question)
            if not target_loan and self.context.get('last_action') == 'listed_loan_types':
                for loan in loans:
                    if loan.loanType.lower() in query.lower():
                        target_loan = loan
                        break
            
            # Still no match - suggest available options
            if not target_loan:
                return self._suggest_loan_types(loans)
            
            # Update context and format response
            self.context['previous_loan_type'] = target_loan.loanType
            self.context['last_action'] = 'provided_loan_details'
            
            return self._format_loan_details(target_loan, query)
        except Exception as e:
            logger.error(f"Error getting loan details: {str(e)}")
            return "Unable to retrieve loan details at this time."
    
    def _suggest_loan_types(self, loans) -> str:
        """Suggest available loan types when none is specified"""
        loan_names = [loan.loanType for loan in loans]
        return (
            f"Please specify which loan you're interested in. "
            f"We offer: {', '.join(loan_names)}. "
            f"For example: 'What's the interest rate for {loan_names[0]}?'"
        )
    
    def _format_loan_details(self, loan: Loans, query: str) -> str:
        """Format loan details based on what was asked"""
        query_lower = query.lower()
        response = f"{loan.loanType} Details:\n"
        
        # Handle specific attribute queries
        if "minimum" in query_lower and "amount" in query_lower:
            response += f"- Minimum amount: NPR {loan.minAmount:,.2f}\n"
        elif "maximum" in query_lower and "amount" in query_lower:
            response += f"- Maximum amount: NPR {loan.maxAmount:,.2f}\n"
        elif "term" in query_lower or "duration" in query_lower:
            response += f"- Term: {loan.minTerm} to {loan.maxTerm} months\n"
        
        # Always include interest rate if not already mentioned
        if "interest" not in query_lower and "rate" not in query_lower:
            response += f"- Interest rate: {loan.interestRate}%\n"
        
        # Include description if asking general info
        if any(word in query_lower for word in ['about', 'details', 'information']):
            response += f"- Description: {loan.description}\n"
        
        response += f"\nWhat else would you like to know about {loan.loanType}?"
        return response
    
    def _get_user_loan_details(self) -> str:
        """Get loan details for the authenticated user"""
        if not self.user_id:
            return "Please log in to view your loan details."
            
        try:
            account = Account.objects.get(user__id=self.user_id)
            if not account.loanID:
                return self._suggest_loan_types(Loans.objects.all())
                
            loan = account.loanID
            return (
                f"Your {loan.loanType} Loan Details:\n"
                f"- Amount: NPR {loan.loanAmount:,.2f}\n"
                f"- Outstanding: NPR {loan.outstandingAmount:,.2f}\n"
                f"- Interest Rate: {loan.interestRate}%\n"
                f"- Term: {loan.loanTerm} months"
            )
        except Exception as e:
            logger.error(f"Error getting user loan details: {str(e)}")
            return "Unable to retrieve your loan details at this time."
    
    def _handle_transaction_query(self) -> str:
        """Handle transaction history queries"""
        if not self.user_id:
            return "Please log in to view your transaction history."
            
        try:
            account = Account.objects.get(user__id=self.user_id)
            transactions = Transactions.objects.filter(
                Q(account=account) & Q(status='completed')
            )
            return self._format_transactions(transactions)
        except Exception as e:
            logger.error(f"Transaction query error: {str(e)}")
            return "Unable to retrieve transactions at this time."
    
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
        """Format transaction data for response"""
        if not transactions.exists():
            return "No recent transactions found."
            
        serialized = TransactionsSerializer(transactions, many=True).data
        return "Recent transactions:\n" + "\n".join(
            f"{t['amount']:,.2f} NPR - {t['description']} ({t['date']})"
            for t in serialized
        )
    
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
    
    def process_query(self, query: str) -> BankingResponse:
        """Main entry point for query processing"""
        try:
            query_type = self.classifier.classify(query)
            logger.debug(f"Classified '{query}' as {query_type}")
            
            if query_type == QueryType.DIRECT:
                response = self._handle_database_query(query)
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
            return BankingResponse(
                query=query,
                type=QueryType.STEPS,
                response="I'm a banking assistant. How can I help with your banking needs?",
                confidence=0.9
            )
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            return BankingResponse(
                query=query,
                type=QueryType.STEPS,
                response="I couldn't process your request.",
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