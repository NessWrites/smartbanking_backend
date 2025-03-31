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
import re
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import BaseMessage


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
    context: Optional[dict] = None  # Added context field
    
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
        self._initialize_patterns()
        
    def _initialize_prompts(self):
        """Initialize classification prompts"""
        self.classifier_prompt = PromptTemplate.from_template("""
            Classify this banking query into exactly one category:
            
            DIRECT - For direct information requests:
            - "my balance", "check my balance", "current balance"
            - "my transactions", "transaction history"
            - "my loan details", "loan status"
            - "types of loan", "what loans do you offer"
            - "account statement"
            - "tell me more about [loan type]"
            - "details about [loan type]"
            
            STEPS - For procedural/how-to questions:
            - "how to check balance", "steps to view transactions"
            - "how to apply for loan", "process to transfer money"
            - "what's needed to open account"
            
            CALCULATIONS - For math/financial calculations:
            - "calculate interest on 5000", "compute EMI for 10000"
            - "convert 200 USD to NPR", "what's 5% of 10000"
            
            Also identify if the query is about:
            - loan products (types of loans available)
            - specific loan terms (interest rates, amounts, etc.)
            Respond with ONLY the classification word: direct, steps, or calculations
            
            Query: {query}
            Classification:
        """)

    def _initialize_patterns(self):
        """Initialize pre-compiled regex patterns"""
        # Steps patterns (checked first)
        self.steps_patterns = [
            re.compile(r'how to', re.IGNORECASE),
            re.compile(r'steps to', re.IGNORECASE),
            re.compile(r'way to', re.IGNORECASE),
            re.compile(r'process (to|for)', re.IGNORECASE),
            re.compile(r'what (do|should) i do to', re.IGNORECASE)
        ]
        
        # Direct patterns
        self.direct_patterns = [
            re.compile(r'\b(my|check|view|show)\s+(balance|transactions?|loans?)\b', re.IGNORECASE),
            re.compile(r'\b(account\s+statement|loan\s+status)\b', re.IGNORECASE),
            re.compile(r'\b(send|transfer)\s+money\b', re.IGNORECASE),
            re.compile(r'\btypes?\s+of\s+loans?\b', re.IGNORECASE),
            re.compile(r'\btell\s+me\s+more\s+about\b', re.IGNORECASE),
            re.compile(r'\bdetails?\s+about\b', re.IGNORECASE)
        ]
        
        # Calculation patterns
        self.calc_patterns = [
            re.compile(r'calculat(e|ion)', re.IGNORECASE),
            re.compile(r'comput(e|ation)', re.IGNORECASE),
            re.compile(r'convert', re.IGNORECASE),
            re.compile(r'\d+\s*%\s+of\s+\d+', re.IGNORECASE),
            re.compile(r'emi', re.IGNORECASE)
        ]
        
        # Add OFF_TOPIC patterns
        self.off_topic_patterns = [
        re.compile(r'\b(trump|biden|politics|sports|weather|movie)\b', re.IGNORECASE),
        re.compile(r'^who (is|are)', re.IGNORECASE),
        re.compile(r'^what is', re.IGNORECASE)
        ]
    def _is_banking_related(self, query: str) -> bool:
        banking_keywords = [
            'balance', 'account', 'loan', 'transaction', 'interest', 'emi', 
            'transfer', 'money', 'currency', 'bank', 'deposit', 'payment', 'withdrawls', 'foreign exchange', 'currency'
        ]
        return any(keyword in query.lower() for keyword in banking_keywords)


    def classify(self, query: str) -> QueryType:
        """Classify query with robust error handling"""
        if not self._is_banking_related(query):
            raise ValueError("This query is not related to banking.")
        try:
            # First try pattern matching
            query_type = self._pattern_match(query)
            if query_type:
                return query_type
                
            # Fall back to LLM if patterns don't match
            return self._llm_classify(query)
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return QueryType.STEPS  # Default fallback
    
    def _pattern_match(self, query: str) -> Optional[QueryType]:
        """Safe pattern matching with pre-compiled regex"""
        try:
            # Check steps patterns first
            if any(pattern.search(query) for pattern in self.steps_patterns):
                return QueryType.STEPS
                
            # Then check direct patterns
            if any(pattern.search(query) for pattern in self.direct_patterns):
                return QueryType.DIRECT
                
            # Finally check calculation patterns
            if any(pattern.search(query) for pattern in self.calc_patterns):
                return QueryType.CALCULATIONS
                
            return None
            
        except Exception as e:
            logger.warning(f"Pattern matching error: {str(e)}")
            return None
    
    def _llm_classify(self, query: str) -> QueryType:
        """Safe LLM classification with fallback"""
        try:
            chain = self.classifier_prompt | self.llm | StrOutputParser()
            raw_output = chain.invoke({"query": query})
            clean_output = self._clean_output(raw_output)
            return QueryType(clean_output)
        except Exception as e:
            logger.warning(f"LLM classification failed: {str(e)}")
            return QueryType.STEPS
    
    def _clean_output(self, raw: str) -> str:
        """Robust output cleaning"""
        try:
            clean = raw.strip().lower()
            clean = re.sub(r'[^a-z]', '', clean.split()[0])  # Take first word
            
            if clean.startswith(('dir', 'dat')):
                return 'direct'
            if clean.startswith(('ste', 'how', 'proc')):
                return 'steps'
            if clean.startswith(('cal', 'com', 'mat')):
                return 'calculations'
                
            return 'steps'  # Default fallback
        except:
            return 'steps'

class ContextAwareBankingAssistant:
    def __init__(self, llm: LlamaCpp, user_id: int = None, max_history: int = 10):
        self.llm = llm
        self.user_id = user_id
        self.max_history = max_history
        self.message_history = ChatMessageHistory()
        self.conversation_context = {
            'last_topics': [],
            'current_focus': None,
            'pending_actions': []
        }
        
    def _analyze_conversation_context(self) -> Dict:
        """Analyze the entire conversation history to extract context"""
        context = {
            'mentioned_products': [],
            'ongoing_requests': [],
            'user_preferences': {}
        }
        
        # Analyze last 5 messages for context
        recent_messages = [msg.content for msg in self.message_history.messages[-5:]]
        
        # Detect mentioned banking products
        banking_products = ['loan', 'account', 'card', 'deposit', 'investment']
        for product in banking_products:
            if any(product in msg.lower() for msg in recent_messages):
                context['mentioned_products'].append(product)
        
        # Detect ongoing requests
        question_words = ['how', 'what', 'when', 'where', 'why']
        if any(msg.endswith('?') for msg in recent_messages):
            context['ongoing_requests'] = recent_messages[-1]  # Last question
        
        return context

    def _generate_contextual_response(self, query: str) -> str:
        """Generate response using full conversation context"""
        current_context = self._analyze_conversation_context()
        
        # Prepare context prompt
        context_prompt = f"""
        Conversation Context:
        - Recent Topics: {', '.join(current_context['mentioned_products'])}
        - Ongoing Requests: {current_context['ongoing_requests']}
        
        Current Query: {query}
        
        Instructions:
        1. Maintain natural conversation flow
        2. Reference previous topics when relevant
        3. If this is a follow-up question, answer accordingly
        4. Be concise but helpful
        """
        
        # Get base response from appropriate tool
        base_response = self._route_to_tool(query)
        
        # Enhance with contextual understanding
        enhanced_prompt = f"""
        {context_prompt}
        
        Base Response: {base_response}
        
        Enhance this response to be more contextual and natural:
        """
        
        return self.llm.invoke(enhanced_prompt)

    def _route_to_tool(self, query: str) -> str:
        """Route query to appropriate tool with context awareness"""
        context = self._analyze_conversation_context()
        
        # Check if this is a follow-up to previous question
        if context['ongoing_requests']:
            last_question = context['ongoing_requests'][-1]
            if self._is_follow_up(query, last_question):
                return self._handle_follow_up(query, last_question)
        
        # Normal routing logic
        query_type = self.classifier.classify(query)
        
        if query_type == QueryType.DIRECT:
            return self._fetch_from_database(query)
        elif query_type == QueryType.STEPS:
            return self._generate_steps_response(query)
        else:
            return self.agent.invoke({"input": query})["output"]

    def _is_follow_up(self, current_query: str, previous_query: str) -> bool:
        """Determine if current query is a follow-up to previous"""
        follow_up_phrases = [
            'what about', 'and', 'also', 'more about', 
            'tell me more', 'explain', 'details',
            'how about', 'what if', 'can i'
        ]
        return any(phrase in current_query.lower() for phrase in follow_up_phrases)

    def _handle_follow_up(self, current_query: str, previous_query: str) -> str:
        """Handle follow-up questions with context"""
        # Get original response
        original_response = self._route_to_tool(previous_query)
        
        # Generate contextual follow-up
        prompt = f"""
        Original Question: {previous_query}
        Original Response: {original_response}
        
        Follow-up Question: {current_query}
        
        Provide a detailed, contextual answer that builds on the original response:
        """
        
        return self.llm.invoke(prompt)



class BankingAssistant:
    def __init__(self, llm: LlamaCpp, user_id: int = None, max_history: int = 10):
        self.llm = llm
        self.user_id = user_id
        self.max_history = max_history
        self.message_history = ChatMessageHistory()
        self.context = {}  # Track conversation context
        
        # Initialize components
        self.classifier = QueryClassifier(llm)
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent()
    
    def _initialize_tools(self) -> list:
        return [
            Tool(
                name="AccountInformation",
                func=self._fetch_from_database,
                description=(
                    "Useful for account balance, transactions, and loan details and other product information including:"
                    "- Account balances and transactions "
                    "- Loan details and status "
                    "- Available loan products and their terms "
                    "- Interest rates and eligibility criteria "
                    "Example queries: "
                    "'what types of loans do you offer' "
                    "'what's the interest rate for home loans' "
                    "'minimum amount for business loan' "
                    "'tell me more about personal loans'"
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
            query_lower = query.lower()
            account = Account.objects.get(user__id=self.user_id)
            
            if "balance" in query_lower:
                try:
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
                # User-specific loan details
                if any(phrase in query_lower for phrase in ["my", "mine", "status", "details"]):
                    print("user_loan details")
                    return self._get_loan_details(account)
                
                # General loan queries (types or specific details)
                if any(phrase in query_lower for phrase in ['types of loan', 'loan types', 
                                                            'what loans', 'available loans', 'loan services']):
                    print("Types of loan")
                    return self._get_all_loan_types()
                
                # Specific loan details (e.g., "minimum amount for personal loan")
                return self._get_specific_loan_details(query)
            
            return "I couldn't find that information. Please try being more specific."
        
        except Account.DoesNotExist:
            logger.error("Account not found for user")
            return "Account not found. Please contact support."
        except Exception as e:
            logger.error(f"Database access error: {str(e)}")
            return "Unable to fetch data at this time."
            
    def _get_all_loan_types(self) -> str:
        """Retrieve all available loan types"""
        loans = Loans.objects.all()
        if not loans.exists():
            return "No loan products currently available."
        
        loan_list = "\n".join(
            f"{loan.loanType} ({loan.interestRate}% interest)" 
            for loan in loans
        )
        
        # Store context that we just showed loan types
        self.context['last_action'] = 'listed_loan_types'
        return f"Available loan types:\n{loan_list}\n\nAsk about specific loans for more details."
    
    def _get_specific_loan_details(self, query: str) -> str:
        """Get details for a specific loan type, answering the specific question asked"""
        query_lower = query.lower()
        loans = Loans.objects.all()
        target_loan = None
    
        # Identify the loan type
        for loan in loans:
            all_names = [loan.loanType.lower()]
            if loan.common_names:
                all_names.extend([name.strip().lower() for name in loan.common_names.split(',')])
            if any(name in query_lower for name in all_names):
                target_loan = loan.loanType
                break
    
        # Check if this is a follow-up after listing loans
        if not target_loan and 'last_action' in self.context and self.context['last_action'] == 'listed_loan_types':
            for loan in loans:
                if loan.loanType.lower() in query_lower:
                    target_loan = loan.loanType
                    break
    
        if not target_loan:
            return "Could not identify the loan type. Please specify which loan you're interested in."
    
        try:
            loan = Loans.objects.get(loanType__iexact=target_loan)
            self.context['previous_loan_type'] = loan.loanType
            self.context['last_action'] = 'provided_loan_details'
    
            # Base response
            response = f"{loan.loanType} Details: "
    
            # Answer specific attribute if asked
            if "minimum" or "min" in query_lower and "amount" or "loan amount" in query_lower:
                response += f"The minimum amount for {loan.loanType} is NPR {loan.minAmount:,.2f} "
            elif "maximum" in query_lower and "amount" in query_lower:
                response += f"The maximum amount for {loan.loanType} is NPR {loan.maxAmount:,.2f} "
            elif "interest" in query_lower or "rate" in query_lower:
                response += f"The interest rate for {loan.loanType} is {loan.interestRate}% "
            elif "term" in query_lower or "duration" in query_lower:
                if "minimum" in query_lower:
                    response += f"The minimum term for {loan.loanType} is {loan.minTerm} months "
                elif "maximum" in query_lower:
                    response += f"The maximum term for {loan.loanType} is {loan.maxTerm} months "
                else:
                    response += f"Terms range from {loan.minTerm} to {loan.maxTerm} months "
            elif "description" in query_lower or "about" in query_lower:
                response += f"Description: {loan.description} "
            else:
                # Default to full details if no specific attribute is asked
                response += (
                    f"Description: {loan.description}\n"
                    f"Interest Rate: {loan.interestRate}%\n"
                    f"Minimum Amount: NPR {loan.minAmount:,.2f}\n"
                    f"Maximum Amount: NPR {loan.maxAmount:,.2f}\n"
                    f"Minimum Term: {loan.minTerm} months\n"
                    f"Maximum Term: {loan.maxTerm} months\n"
                )
    
            # Add follow-up prompt unless already full details
            if not ("description" in query_lower or "about" in query_lower) and not all(attr in query_lower for attr in ["minimum", "maximum", "interest", "term"]):
                response += f"with an interest rate of {loan.interestRate}%. "
                response += f"The maximum loan amount is NPR {loan.maxAmount:,.2f}. "
                response += f"What more would you like to know about {loan.loanType}?"
    
            return response.strip()
    
        except Loans.DoesNotExist:
            return f"Could not find details for {target_loan}."
    
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
            return "You don't have any active loans. Would you like to know more about loans?"
            
        try:
            loan = Loans.objects.get(pk=account.loanID.pk)
            data = LoansSerializer(loan).data
            
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
        try:
            query_type = self.classifier.classify(query)
            logger.debug(f"Classified '{query}' as {query_type}")
            if query_type == QueryType.DIRECT:
                response = self._fetch_from_database(query)
            elif query_type == QueryType.STEPS:
                response = self._generate_steps_response(query)
            else:
                response = self.agent.invoke({"input": query})["output"]
            self._update_chat_history(query, response)
            return BankingResponse(query=query, type=query_type, response=response, confidence=0.9, context=self.context.copy())
        except ValueError as ve:
            return BankingResponse(
                query=query,
                type=QueryType.STEPS,
                response="I'm a banking assistant and can only help with banking-related questions. How can I assist you with your banking needs?",
                confidence=0.9
            )
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            return BankingResponse(query=query, type=QueryType.STEPS, response="I couldn't process your request.", confidence=0.1)
        
    def _update_chat_history(self, query: str, response: str):
        """Maintain conversation history and update context"""
        self.message_history.add_messages([
            HumanMessage(content=query),
            AIMessage(content=response)
        ])
        
        # Trim history if needed
        if len(self.message_history.messages) > self.max_history:
            self.message_history.messages = self.message_history.messages[-self.max_history:]
            
        # Update context based on the conversation
        self._update_conversation_context(query, response)
    
    def _update_conversation_context(self, query: str, response: str):
        """Update the conversation context based on the current exchange"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Detect if we just listed loan types
        if 'available loan types:' in response:
            self.context['last_action'] = 'listed_loan_types'
        
        # Detect if user is asking about a specific loan type
        elif any(phrase in query_lower for phrase in ['tell me more about', 'details about']):
            # Try to extract the loan type from the query
            for loan in Loans.objects.all():
                if loan.loanType.lower() in query_lower:
                    self.context['previous_loan_type'] = loan.loanType
                    self.context['last_action'] = 'asked_about_loan'
                    break