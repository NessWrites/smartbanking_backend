# utils/query_processor.py
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from .serializers import AccountSerializer, LoansSerializer
from .models import Account, Loans, User
from decimal import Decimal
import logging
from typing import Dict, Any, Optional
import re
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)


class QueryProcessor:
    def __init__(self):
        # Initialize Llama 3 model for CPU-only usage
        self.model_path = "/Users/ness/Islington/unsloth.Q4_K_M.gguf"
        self.llm = None
        self.steps_llm = None  # Separate instance for steps queries
        
    async def initialize(self):
        """Async initialization"""
        try:
            # Common configuration for both models
            common_config = {
                "model_path": self.model_path,
                "n_ctx": 2048,
                "n_threads": 4,
                "n_gpu_layers": 0,  # CPU-only
                "verbose": False,
                "n_batch": 64  # Explicitly set to avoid warnings
            }
            
        # Main model configuration
            self.llm = LlamaCpp(
                temperature=0.7,
                **common_config
            )
          # Fine-tuned model specifically for steps
        # Fine-tuned model for steps
            self.steps_llm = LlamaCpp(
                temperature=0.7,
                **common_config
            )
            
            # Warm up both models
            await sync_to_async(self.llm.__call__)("warmup")
            await sync_to_async(self.steps_llm.__call__)("warmup")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError("Failed to initialize language models")

        # Define response schemas
        self.response_schemas = [
            ResponseSchema(name="query_type", description="Type of query: 'direct', 'steps', or 'calculation'"),
            ResponseSchema(name="entity", description="Entity being queried: 'account','transfer', 'loan', 'transaction', 'withdraw', 'exchange'"),
            ResponseSchema(name="attribute", description="Specific attribute requested", optional=True),
            ResponseSchema(name="make_transfer", description="make a transfer", optional=True),
            ResponseSchema(name="amount", description="Amount for calculations", optional=True),
            ResponseSchema(name="time_period", description="Time period in years", optional=True),
            ResponseSchema(name="from_currency", description="Source currency", optional=True),
            ResponseSchema(name="to_currency", description="Target currency", optional=True)
        ]   
        
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        
        # Enhanced prompt with clear pattern examples
        self.intent_prompt = ChatPromptTemplate.from_template("""
            Classify this banking query based on these patterns:
            
            DIRECT QUERIES (immediate information):
            - "my balance", "check my balance", "tell me my balance" → {{"query_type": "direct", "entity": "account", "attribute": "balance"}}
            - "my loan", "check my loan", "tell me my loan" → {{"query_type": "direct", "entity": "loan", "attribute": "details"}}
            - "my transactions", "last 5 transactions" → {{"query_type": "direct", "entity": "transaction", "attribute": "history"}}
            
            STEPS QUERIES (how-to/guidance):
            - "how to check my balance", "steps to check balance" → {{"query_type": "steps", "entity": "account", "attribute": "balance"}}
            - "how to check my loan", "steps to get loan" → {{"query_type": "steps", "entity": "loan", "attribute": "application"}}
            - "how to withdraw money" → {{"query_type": "steps", "entity": "withdraw", "attribute": "procedure"}}
            - "how to transfer money" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}
            - "how to transfer my balance" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}
            - "how to transfer balance" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}
            - "steps to send balance" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}
            - "how to move funds" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}
            
            CALCULATION QUERIES (math operations):
            - "calculate interest for 5000" → {{"query_type": "calculation", "entity": "loan", "calculation_type": "interest", "amount": "5000"}}
            - "convert 200 USD to NPR" → {{"query_type": "calculation", "entity": "exchange", "calculation_type": "exchange", "amount": "200", "from_currency": "USD", "to_currency": "NPR"}}
            
            Current Query: {input}
            
            Respond ONLY with valid JSON:
            {format_instructions}
        """)

        # Create the classification chain
        self.classification_chain = (
            {"input": RunnablePassthrough(), "format_instructions": lambda x: self.output_parser.get_format_instructions()}
            | self.intent_prompt
            | self.llm
            | self.output_parser
        )

    async def classify_intent(self, query: str) -> Dict[str, Any]:
        """Enhanced intent classification with transfer detection"""
        # First check for explicit transfer phrases
        transfer_phrases = ["transfer balance", "move funds", "send money", "send balance"]
        if any(phrase in query.lower() for phrase in transfer_phrases):
            return {
                "query_type": "steps",
                "entity": "transaction",
                "attribute": "transfer"
            }
            
        """Classify query intent with pattern matching fallback"""
        try:
            # First try pattern matching for faster response
            if self._is_steps_query(query):
                return await self._pattern_match_steps(query)
            elif self._is_direct_query(query):
                return await self._pattern_match_direct(query)
            elif self._is_calculation_query(query):
                return await self._pattern_match_calculation(query)
            
            # Fall back to LLM if pattern matching fails
            result = await self.classification_chain.ainvoke(query)
            return self._normalize_output(result)
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {
                "query_type": "direct",
                "entity": "account",
                "attribute": "balance"
            }

    def _is_steps_query(self, query: str) -> bool:
        """Check if query is asking for steps"""
        steps_phrases = [
            "how to", "steps to", "way to", "process to",
            "procedure for", "guide me", "show me how",
            "tell me how", "explain how", "walk me through"
        ]
        return any(phrase in query.lower() for phrase in steps_phrases)

    def _is_direct_query(self, query: str) -> bool:
        """Check if query is direct information request"""
        direct_phrases = [
            "my", "check", "tell me", "show me",
            "what is", "what's", "give me",
            "current", "latest", "view", "see my"
        ]
        return any(phrase in query.lower() for phrase in direct_phrases)

    def _is_calculation_query(self, query: str) -> bool:
        """Check if query requires calculation"""
        calc_phrases = [
            "calculate", "compute", "convert",
            "how much", "what would be", "total"
        ]
        return any(phrase in query.lower() for phrase in calc_phrases)

    

    async def _pattern_match_steps(self, query: str) -> Dict[str, Any]:
        """Enhanced pattern matching for steps queries with clear priority"""
        query_lower = query.lower()
        
        # 1. First check for transfer requests (highest priority)
        transfer_patterns = [
            (r"(how to|steps to|way to) (transfer|send|move) (?:my|your)? (?:money|funds|balance)", 
             "transaction", "transfer"),
            (r"(?:can you|please) (?:show|explain) (?:how|the steps) to (?:transfer|send)", 
             "transaction", "transfer"),
            (r"(?:i want|i need) to (?:transfer|send|move) (?:money|funds|balance)", 
             "transaction", "transfer"),
            (r"process (?:for|to) (?:transferring|sending) (?:money|funds)", 
             "transaction", "transfer")
        ]
        
        for pattern, entity, attribute in transfer_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return {
                    "query_type": "steps",
                    "entity": entity,
                    "attribute": attribute
                }
        
        # 2. Then check for balance inquiries
        balance_phrases = [
            r"how to (?:check|view|see) (?:my|your)? (?:balance|account balance)",
            r"steps to (?:check|view) (?:my )?balance",
            r"way to (?:see|check) (?:my )?(?:current )?balance",
            r"(?:show|display) (?:me )?(?:my )?balance"
        ]
        
        for phrase in balance_phrases:
            if re.search(phrase, query_lower):
                return {
                    "query_type": "steps",
                    "entity": "account",
                    "attribute": "balance"
                }
        
        # 3. Other transaction types
        transaction_patterns = {
            r"(?:how to|steps to) (?:withdraw|get) cash(?: from atm)?": 
                {"entity": "withdraw", "attribute": "atm"},
            r"(?:process|procedure) for (?:withdrawing|getting) money": 
                {"entity": "withdraw", "attribute": "procedure"},
            r"(?:how to|steps to) (?:deposit|add) (?:money|funds)": 
                {"entity": "transaction", "attribute": "deposit"},
            r"(?:how to|steps to) (?:exchange|convert) currency": 
                {"entity": "exchange", "attribute": "currency"},
            r"(?:how to|steps to) (?:pay|make payment)": 
                {"entity": "transaction", "attribute": "payment"}
        }
        
        for pattern, entity_info in transaction_patterns.items():
            if re.search(pattern, query_lower):
                return {"query_type": "steps", **entity_info}
        
        # 4. Loan-related queries
        loan_patterns = {
            r"(?:how to|steps to) (?:apply for|get) (?:a|my) loan": 
                {"entity": "loan", "attribute": "application"},
            r"(?:process|steps) for (?:loan|credit) application": 
                {"entity": "loan", "attribute": "procedure"},
            r"(?:how to|steps to) check (?:loan|credit) status": 
                {"entity": "loan", "attribute": "status"}
        }
        
        for pattern, entity_info in loan_patterns.items():
            if re.search(pattern, query_lower):
                return {"query_type": "steps", **entity_info}
        
        # 5. Fallback to general account help
        return {
            "query_type": "steps",
            "entity": "account",
            "attribute": "general_help"
        }
    
    async def _pattern_match_direct(self, query: str) -> Dict[str, Any]:
        """Pattern matching for direct queries"""
        query_lower = query.lower()
        patterns = {
            r"(balance|account)": {"entity": "account", "attribute": "balance"},
            r"(loan|credit)": {"entity": "loan", "attribute": "details"},
            r"(transaction|history|statement)": {"entity": "transaction", "attribute": "history"},
            r"(withdraw|withdrawal|limit)": {"entity": "withdraw", "attribute": "limit"}
        }
        
        for pattern, entity_info in patterns.items():
            if re.search(pattern, query_lower):
                return {"query_type": "direct", **entity_info}
        
        return {"query_type": "direct", "entity": "account", "attribute": "balance"}

    async def _pattern_match_calculation(self, query: str) -> Dict[str, Any]:
        """Pattern matching for calculation queries"""
        query_lower = query.lower()
        
        # Interest calculation
        interest_match = re.search(r"(interest|rate).*?(\d+)", query_lower)
        if interest_match:
            return {
                "query_type": "calculation",
                "entity": "loan",
                "calculation_type": "interest",
                "amount": interest_match.group(2),
                "time_period": "1"  # Default to 1 year
            }
        
        # Currency conversion
        exchange_match = re.search(r"convert (\d+) ([A-Z]{3}) to ([A-Z]{3})", query_lower, re.IGNORECASE)
        if exchange_match:
            return {
                "query_type": "calculation",
                "entity": "exchange",
                "calculation_type": "exchange",
                "amount": exchange_match.group(1),
                "from_currency": exchange_match.group(2).upper(),
                "to_currency": exchange_match.group(3).upper()
            }
        
        return {"query_type": "calculation", "entity": "loan", "calculation_type": "interest"}

    def _normalize_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure consistent output format"""
        return {
            "query_type": output.get("query_type", "direct").lower(),
            "entity": output.get("entity", "account").lower(),
            "attribute": output.get("attribute", "").lower(),
            "amount": output.get("amount"),
            "time_period": output.get("time_period"),
            "from_currency": output.get("from_currency"),
            "to_currency": output.get("to_currency")
        }

    async def process_query_async(self, user, query):
        """Async version of process_query"""
        try:
            classification = await self.classify_intent(query)
            
            if classification["query_type"] == "direct":
                return await self.handle_direct_query(user, classification)
            elif classification["query_type"] == "steps":
                return await self.generate_steps_response(classification)
            elif classification["query_type"] == "calculation":
                return await self.handle_calculation_query(user, classification)
            return "Please rephrase your request."
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "Sorry, I couldn't process your request."


    async def handle_direct_query(self, user: User, classification: Dict[str, Any]) -> str:
        """Handle direct information requests"""
        try:
            account = await Account.objects.aget(user=user)
            serializer = AccountSerializer(account)
            
            entity = classification["entity"]
            attribute = classification["attribute"]
            
            if entity == "account":
                if attribute == "balance":
                    return f"Your current balance is: {serializer.data['balance']}"
                elif attribute == "details":
                    return str(serializer.data)
                elif attribute == "transactions":
                    return "Your recent transactions: [...]"  # Implement actual logic
                    
            elif entity == "loan":
                if not account.loanID:
                    return "You don't have an active loan."
                
                loan = await Loans.objects.aget(pk=account.loanID.pk)
                loan_serializer = LoansSerializer(loan)
                
                if attribute == "details":
                    return str(loan_serializer.data)
                elif attribute == "balance":
                    return f"Your loan balance is: {loan_serializer.data['outstandingAmount']}"
                    
            elif entity == "transaction":
                return "Your transaction history: [...]"  # Implement actual logic
                
            return f"I can provide information about your {entity}. Please be more specific."
            
        except Account.DoesNotExist:
            return "Account not found. Please contact customer support."
        except Exception as e:
            logger.error(f"Direct query error: {str(e)}")
            return "Sorry, I couldn't retrieve that information."

    async def generate_steps_response(self, classification: Dict[str, Any]) -> str:
        """Generate step-by-step instructions using fine-tuned model"""
        entity = classification["entity"]
        attribute = classification["attribute"]
        
        # Create a specialized prompt for steps
        steps_prompt = ChatPromptTemplate.from_template("""
            You are a banking assistant specialized in providing clear, step-by-step instructions.
            Provide detailed steps to help the user with this request:
            
            Entity: {entity}
            Action: {attribute}
            
            Instructions:
            1. Be extremely clear and concise
            2. Number each step
            3. Include any important warnings or notes
            4. Use simple language
            
            Request: {query}
            
            Respond ONLY with the numbered steps:
        """)
        
        chain = (
            {"entity": lambda x: entity,
             "attribute": lambda x: attribute,
             "query": RunnablePassthrough()}
            | steps_prompt
            | self.steps_llm #invoking fined tune model
        )
        
        # Generate steps using the fine-tuned model
        result = await chain.ainvoke(classification)
        return str(result)

    async def handle_calculation_query(self, user: User, classification: Dict[str, Any]) -> str:
        """Perform financial calculations"""
        try:
            calc_type = classification.get("calculation_type", "").lower()
            
            if calc_type == "interest":
                return await self.calculate_interest(user, classification)
            elif calc_type == "exchange":
                return self.calculate_exchange(classification)
            else:
                return "I can perform interest or currency calculations. Please specify which one you need."
        except Exception as e:
            logger.error(f"Calculation error: {str(e)}")
            return "Couldn't perform the calculation. Please check your request."

    async def calculate_interest(self, user: User, classification: Dict[str, Any]) -> str:
        """Calculate loan interest"""
        try:
            account = await Account.objects.aget(user=user)
            if not account.loanID:
                return "You don't have an active loan."
            
            loan = account.loanID
            principal = Decimal(classification.get("amount", str(account.balance)))
            years = Decimal(classification.get("time_period", "1"))
            
            interest = (principal * loan.interestRates * years) / 100
            return (f"Interest calculation:\n"
                   f"Principal: {principal}\n"
                   f"Rate: {loan.interestRates}%\n"
                   f"Time: {years} year(s)\n"
                   f"Total Interest: {interest:.2f}")
                   
        except Exception as e:
            logger.error(f"Interest calculation error: {str(e)}")
            return "Couldn't calculate interest. Please check your numbers."

    def calculate_exchange(self, classification: Dict[str, Any]) -> str:
        """Calculate currency conversion"""
        try:
            amount = Decimal(classification.get("amount", "1"))
            from_curr = classification.get("from_currency", "USD").upper()
            to_curr = classification.get("to_currency", "NPR").upper()
            
            # In production, replace with actual API call
            rates = {
                "USD_NPR": Decimal("133.50"),
                "EUR_NPR": Decimal("145.25"),
                "GBP_NPR": Decimal("170.80")
            }
            
            rate_key = f"{from_curr}_{to_curr}"
            if rate_key not in rates:
                return f"No exchange rate available for {from_curr} to {to_curr}"
                
            converted = amount * rates[rate_key]
            return (f"Currency Conversion:\n"
                   f"Amount: {amount} {from_curr}\n"
                   f"Rate: 1 {from_curr} = {rates[rate_key]} {to_curr}\n"
                   f"Result: {converted:.2f} {to_curr}")
                   
        except Exception as e:
            logger.error(f"Exchange calculation error: {str(e)}")
            return "Couldn't perform currency conversion. Please check your request."
        
    