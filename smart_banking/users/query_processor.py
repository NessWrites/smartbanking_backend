# utils/query_processor.py
import json
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
        self.model_path = "/Users/ness/Islington/unsloth.Q4_K_M.gguf"
        self.llm = None
        self.steps_llm = None
        
    async def initialize(self):
        """Initialize models"""
        try:
            common_config = {
                "model_path": self.model_path,
                "n_ctx": 2048,
                "n_gpu_layers": 0,
                "verbose": False,
                "n_batch": 64
            }
            self.llm = LlamaCpp(temperature=0.3, **common_config)
            self.steps_llm = LlamaCpp(temperature=0.3, **common_config)
            
            await sync_to_async(self.llm.invoke)("warmup")
            await sync_to_async(self.steps_llm.invoke)("warmup")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError("Failed to initialize language model")
        
                # Updated response schemas - only required fields
        self.response_schemas = [
            ResponseSchema(name="query_type", description="Type of query: 'direct', 'steps', or 'calculation'"),
            ResponseSchema(name="entity", description="Entity being queried: 'account','transfer', 'loan', 'transaction', 'withdraw', 'exchange'")
        ]
        
        # Optional fields (only present when relevant)
        self.optional_schemas = [
            ResponseSchema(name="attribute", description="Specific attribute requested", optional=True),
            ResponseSchema(name="calculation_type", description="For calculations: 'interest', 'emi', 'repayment' etc", optional=True),
            ResponseSchema(name="amount", description="Amount for calculations", optional=True),
            ResponseSchema(name="time_period", description="Time period in years", optional=True),
            ResponseSchema(name="from_currency", description="Source currency", optional=True),
            ResponseSchema(name="to_currency", description="Target currency", optional=True)
        ]
         # Combine both required and optional schemas
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.response_schemas + self.optional_schemas
        )

        
        # Enhanced prompt with clear pattern examples
        self.intent_prompt = ChatPromptTemplate.from_template("""
            Classify this banking query based on these patterns:
            
            DIRECT QUERIES (immediate information):
            - "my balance", "check my balance", "tell me my balance" → {{"query_type": "direct", "entity": "account", "attribute": "balance"}}
            - "my loan details", "check my loan status", "do i have any loans" → {{"query_type": "direct", "entity": "loan", "attribute": "details"}}
            - "my transactions", "last 5 transactions" → {{"query_type": "direct", "entity": "transaction", "attribute": "history"}}
            - "current exchange rate for USD" → {{"query_type": "direct", "entity": "exchange", "attribute": "rate"}}
            
            STEPS QUERIES (how-to/guidance):
            - "how to check my balance" → {{"query_type": "steps", "entity": "account", "attribute": "balance"}}
            - "how to apply for loan" → {{"query_type": "steps", "entity": "loan", "attribute": "application"}}
            - "what are the types of loan" → {{"query_type": "steps", "entity": "loan", "attribute": "application"}}
            - "steps to transfer money" → {{"query_type": "steps", "entity": "transfer", "attribute": "procedure"}}
            - "how to view transaction history" → {{"query_type": "steps", "entity": "transaction", "attribute": "history"}}
            - "how to withdraw money" → {{"query_type": "steps", "entity": "withdraw", "attribute": "procedure"}}
            - "how to transfer money" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}
            - "how to transfer my balance" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}
            - "how to transfer balance" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}
            - "steps to send balance" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}
            - "how to move funds" → {{"query_type": "steps", "entity": "transfer", "attribute": "make_transfer"}}

            
            CALCULATION QUERIES (math operations):
            - "calculate interest for 5000 loan" → {{"query_type": "calculation", "entity": "loan", "calculation_type": "interest", "amount": "5000"}}
            - "what would be my EMI for 10000" → {{"query_type": "calculation", "entity": "loan", "calculation_type": "emi", "amount": "10000"}}
            - "convert 200 USD to NPR" → {{"query_type": "calculation", "entity": "exchange", "calculation_type": "exchange", "amount": "200", "from_currency": "USD", "to_currency": "NPR"}}
            - "calculate total repayment for 50000 loan" → {{"query_type": "calculation", "entity": "loan", "calculation_type": "repayment", "amount": "50000"}}
            
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

        # Steps prompt for generating step-by-step instructions
        self.steps_prompt = ChatPromptTemplate.from_template("""
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

    async def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent with proper prompt handling"""
        print(f"\n=== CLASSIFICATION START ===\nQuery: '{query}'")

        try:
            # First try pattern matching
            print("\n[1] Trying pattern matching...")
            if self._is_steps_query(query):
                print("Pattern matched as STEPS query")
                result = await self._pattern_match_steps(query)
            elif self._is_direct_query(query):
                print("Pattern matched as DIRECT query")
                result = await self._pattern_match_direct(query)
            elif self._is_calculation_query(query):
                print("Pattern matched as CALCULATION query")
                result = await self._pattern_match_calculation(query)
            else:
                print("No pattern match - using LLM classification")

                # Use the intent prompt that we know exists (self.intent_prompt)
                chain_input = {
                    "input": query,
                    "format_instructions": self.output_parser.get_format_instructions()
                }
                result = await self.classification_chain.ainvoke(chain_input)

            print("\n[3] Classification result (pre-normalization):")
            print(json.dumps(result, indent=2))

            normalized = self._normalize_output(result)
            print("\n[4] Normalized output:")
            print(json.dumps(normalized, indent=2))

            return normalized

        except Exception as e:
            print(f"\n[ERROR] Classification failed: {str(e)}")
            print("Returning fallback response (account balance)")
            logger.error(f"Classification error: {str(e)}")
            return {
                "query_type": "steps",
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
        
        # 2. Loan-related queries (high priority)
        loan_patterns = {
            r"\b(types of loans|loan options|kinds of loans)\b": 
                {"entity": "loan", "attribute": "products"},
            r"\b(how to apply|want to apply|get a loan|apply for loan)\b": 
                {"entity": "loan", "attribute": "application"},
            r"\b(how to check|steps to check) (?:my )?loan (?:status|details)\b": 
                {"entity": "loan", "attribute": "check_status"},
            r"\b(how to repay|steps to pay off) (?:my )?loan\b": 
                {"entity": "loan", "attribute": "repayment"},
            r"\b(how to calculate|steps to determine) (?:my )?loan (?:emi|payment)\b": 
                {"entity": "loan", "attribute": "calculation"}
        }
        
        for pattern, entity_info in loan_patterns.items():
            if re.search(pattern, query_lower):
                return {"query_type": "steps", **entity_info}
        
        # 3. Transaction-related queries
        transaction_patterns = {
            r"\b(how to view|steps to see) (?:my )?transaction (?:history|records)\b": 
                {"entity": "transaction", "attribute": "history"},
            r"\b(how to dispute|steps to challenge) (?:a|my) transaction\b": 
                {"entity": "transaction", "attribute": "dispute"},
            r"\b(how to export|steps to download) (?:my )?transactions\b": 
                {"entity": "transaction", "attribute": "export"}
        }
        
        for pattern, entity_info in transaction_patterns.items():
            if re.search(pattern, query_lower):
                return {"query_type": "steps", **entity_info}
        
        # 4. Balance-related queries
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
        
        # 5. Exchange rate queries
        exchange_patterns = {
            r"\b(how to check|steps to find) (?:current|latest) exchange rates\b": 
                {"entity": "exchange", "attribute": "rate_check"},
            r"\b(how to get|steps to obtain) best exchange rate\b": 
                {"entity": "exchange", "attribute": "best_rate"}
        }
        
        for pattern, entity_info in exchange_patterns.items():
            if re.search(pattern, query_lower):
                return {"query_type": "steps", **entity_info}
        
        # Fallback to general account help
        return {
            "query_type": "steps",
            "entity": "account",
            "attribute": "general_help"
        }
    
    async def _pattern_match_direct(self, query: str) -> Dict[str, Any]:
        """Pattern matching for direct queries"""
        query_lower = query.lower()
         # Add this pattern for loan types
        if re.search(r"\b(types of loans|loan options|kinds of loans)\b", query_lower):
            return {
            "query_type": "direct", 
            "entity": "loan",
            "attribute": "types"
                }
        
        # Loan patterns
        loan_patterns = {
            r"\b(my loan|loan balance|do i have a loan|my loan details)\b": 
                {"entity": "loan", "attribute": "details"},
            r"\b(my loan status|loan progress)\b": 
                {"entity": "loan", "attribute": "status"},
            r"\b(my loan payments|loan emi)\b": 
                {"entity": "loan", "attribute": "payments"}
        }
        
        for pattern, entity_info in loan_patterns.items():
            if re.search(pattern, query_lower):
                return {"query_type": "direct", **entity_info}
        
        # Transaction patterns
        transaction_patterns = {
            r"\b(my transactions|transaction history)\b": 
                {"entity": "transaction", "attribute": "history"},
            r"\b(last \d+ transactions)\b": 
                {"entity": "transaction", "attribute": "recent"},
            r"\b(pending transactions)\b": 
                {"entity": "transaction", "attribute": "pending"}
        }
        
        for pattern, entity_info in transaction_patterns.items():
            if re.search(pattern, query_lower):
                return {"query_type": "direct", **entity_info}
        
        # Balance patterns
        balance_patterns = {
            r"\b(my balance|account balance|current balance)\b": 
                {"entity": "account", "attribute": "balance"},
            r"\b(available balance|usable funds)\b": 
                {"entity": "account", "attribute": "available_balance"}
        }
        
        for pattern, entity_info in balance_patterns.items():
            if re.search(pattern, query_lower):
                return {"query_type": "direct", **entity_info}
        
        # Exchange rate patterns
        exchange_patterns = {
            r"\b(current exchange rate for ([A-Z]{3})\b": 
                {"entity": "exchange", "attribute": "rate", "from_currency": lambda m: m.group(1).upper()},
            r"\b(USD to NPR rate|NPR to USD rate)\b": 
                {"entity": "exchange", "attribute": "rate", 
                 "from_currency": lambda m: "USD" if "USD to" in m.group(0) else "NPR",
                 "to_currency": lambda m: "NPR" if "USD to" in m.group(0) else "USD"}
        }
        
        for pattern, entity_info in exchange_patterns.items():
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                processed_info = {}
                for k, v in entity_info.items():
                    if callable(v):
                        processed_info[k] = v(match)
                    else:
                        processed_info[k] = v
                return {"query_type": "direct", **processed_info}
        
        # Fallback to account balance
        return {"query_type": "direct", "entity": "account", "attribute": "balance"}

    async def _pattern_match_calculation(self, query: str) -> Dict[str, Any]:
        """Pattern matching for calculation queries"""
        query_lower = query.lower()
        
        # Loan calculations
        loan_interest_match = re.search(r"(?:calculate|compute) (?:loan )?interest (?:for|on) (\d+)", query_lower)
        if loan_interest_match:
            return {
                "query_type": "calculation",
                "entity": "loan",
                "calculation_type": "interest",
                "amount": loan_interest_match.group(1),
                "time_period": "1"  # Default to 1 year
            }
        
        loan_emi_match = re.search(r"(?:what would be|calculate) (?:my )?emi (?:for|of) (\d+)", query_lower)
        if loan_emi_match:
            return {
                "query_type": "calculation",
                "entity": "loan",
                "calculation_type": "emi",
                "amount": loan_emi_match.group(1),
                "time_period": "5"  # Default to 5 years
            }
        
        # Transaction calculations
        transaction_fee_match = re.search(r"(?:calculate|estimate) (?:transfer|transaction) fee (?:for|on) (\d+)", query_lower)
        if transaction_fee_match:
            return {
                "query_type": "calculation",
                "entity": "transaction",
                "calculation_type": "fee",
                "amount": transaction_fee_match.group(1)
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
        
        # Balance conversion
        balance_conversion_match = re.search(r"(?:convert|calculate) (?:my )?balance from ([A-Z]{3}) to ([A-Z]{3})", query_lower, re.IGNORECASE)
        if balance_conversion_match:
            return {
                "query_type": "calculation",
                "entity": "account",
                "calculation_type": "balance_conversion",
                "from_currency": balance_conversion_match.group(1).upper(),
                "to_currency": balance_conversion_match.group(2).upper()
            }
        
        # Fallback to loan interest calculation
        return {
            "query_type": "calculation",
            "entity": "loan",
            "calculation_type": "interest",
            "amount": "10000",  # Default amount
            "time_period": "1"   # Default to 1 year
        }

    # Update the _normalize_output method to handle missing fields:
    def _normalize_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure consistent output format with debugging"""
        print("Raw classification output:", output)
        
        normalized = {
            "query_type": output.get("query_type", "direct").lower(),
            "entity": output.get("entity", "account").lower()
        }
        
        # Only add fields that exist in the output
        optional_fields = [
            "attribute", "calculation_type", "amount",
            "time_period", "from_currency", "to_currency"
        ]
        
        for field in optional_fields:
            if field in output:
                normalized[field] = output[field].lower() if isinstance(output[field], str) else output[field]
        
        print("Normalized output:", normalized)
        return normalized


    async def process_query_async(self, user, query):
        """Process query based on classification"""
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
                elif attribute == "available_balance":
                    return f"Your available balance is: {serializer.data['balance'] - serializer.data['hold_amount']}"
                elif attribute == "details":
                    return str(serializer.data)
                    
            elif entity == "loan":
                if not account.loanID:
                    return "You don't have an active loan."
                
                loan = await Loans.objects.aget(pk=account.loanID.pk)
                loan_serializer = LoansSerializer(loan)
                
                if attribute == "details":
                    return (f"Loan Details:\n"
                           f"Type: {loan_serializer.data['type']}\n"
                           f"Amount: {loan_serializer.data['loanAmount']}\n"
                           f"Outstanding: {loan_serializer.data['outstandingAmount']}\n"
                           f"Rate: {loan_serializer.data['interestRate']}%")
                elif attribute == "status":
                    return f"Your loan status is: {loan_serializer.data['status']}"
                elif attribute == "payments":
                    return f"Your next payment is: {loan_serializer.data['nextPaymentAmount']} due on {loan_serializer.data['nextPaymentDate']}"
                    
            elif entity == "transaction":
                if attribute == "history":
                    return "Your recent transactions: [...]"  # Implement actual logic
                elif attribute == "recent":
                    return "Your last 5 transactions: [...]"  # Implement actual logic
                elif attribute == "pending":
                    return "Your pending transactions: [...]"  # Implement actual logic
                    
            elif entity == "exchange":
                from_curr = classification.get("from_currency", "USD")
                to_curr = classification.get("to_currency", "NPR")
                rate = self._get_exchange_rate(from_curr, to_curr)
                return f"Current exchange rate: 1 {from_curr} = {rate} {to_curr}"
                
            return f"I can provide information about your {entity}. Please be more specific."
            
        except Account.DoesNotExist:
            return "Account not found. Please contact customer support."
        except Exception as e:
            logger.error(f"Direct query error: {str(e)}")
            return "Sorry, I couldn't retrieve that information."

    async def generate_steps_response(self, classification: Dict[str, Any]) -> str:
        """Generate step-by-step instructions"""
        chain = (
            {"entity": lambda x: classification["entity"],
             "attribute": lambda x: classification["attribute"],
             "query": RunnablePassthrough()}
            | self.steps_prompt
            | self.steps_llm
        )
        
        result = await chain.ainvoke(classification)
        return str(result)

    async def handle_calculation_query(self, user: User, classification: Dict[str, Any]) -> str:
        """Perform financial calculations"""
        try:
            entity = classification["entity"]
            calc_type = classification.get("calculation_type", "").lower()
            
            if entity == "loan":
                return await self.calculate_loan(user, classification)
            elif entity == "transaction":
                return await self.calculate_transaction_fee(classification)
            elif entity == "exchange":
                return await self.calculate_exchange(classification)
            elif entity == "account":
                return await self.calculate_balance_conversion(user, classification)
            else:
                return "I can perform loan, transaction, and currency calculations. Please specify."
        except Exception as e:
            logger.error(f"Calculation error: {str(e)}")
            return "Couldn't perform the calculation. Please check your request."

    async def calculate_loan(self, user: User, classification: Dict[str, Any]) -> str:
        """Enhanced loan calculation handler"""
        try:
            account = await Account.objects.aget(user=user)
            if not account.loanID:
                return "You don't have an active loan."
            
            loan = account.loanID
            calc_type = classification.get("calculation_type", "interest")
            principal = Decimal(classification.get("amount", str(loan.loanAmount)))
            years = Decimal(classification.get("time_period", loan.loanTerm))
            
            if calc_type == "emi":
                # EMI calculation formula: [P x R x (1+R)^N]/[(1+R)^N-1]
                monthly_rate = loan.interestRate / 12 / 100
                months = years * 12
                emi = (principal * monthly_rate * (1+monthly_rate)**months) / ((1+monthly_rate)**months - 1)
                
                return (f"EMI Calculation:\n"
                       f"Principal: {principal}\n"
                       f"Interest Rate: {loan.interestRate}% p.a.\n"
                       f"Term: {years} year(s)\n"
                       f"Monthly EMI: {emi:.2f}")
            else:
                # Standard interest calculation
                interest = (principal * loan.interestRate * years) / 100
                return (f"Interest Calculation:\n"
                       f"Principal: {principal}\n"
                       f"Rate: {loan.interestRate}%\n"
                       f"Time: {years} year(s)\n"
                       f"Total Interest: {interest:.2f}")
                       
        except Exception as e:
            logger.error(f"Loan calculation error: {str(e)}")
            return "Couldn't calculate loan details. Please check your numbers."

    async def calculate_transaction_fee(self, classification: Dict[str, Any]) -> str:
        """Calculate transaction fees"""
        try:
            amount = Decimal(classification.get("amount", "0"))
            fee = amount * Decimal("0.01")  # 1% fee example
            return (f"Transaction Fee Calculation:\n"
                   f"Amount: {amount}\n"
                   f"Fee (1%): {fee:.2f}\n"
                   f"Total Debit: {amount + fee:.2f}")
        except Exception as e:
            logger.error(f"Transaction fee calculation error: {str(e)}")
            return "Couldn't calculate transaction fees. Please check your amount."

    async def calculate_exchange(self, classification: Dict[str, Any]) -> str:
        """Calculate currency conversion"""
        try:
            amount = Decimal(classification.get("amount", "1"))
            from_curr = classification.get("from_currency", "USD").upper()
            to_curr = classification.get("to_currency", "NPR").upper()
            
            rate = self._get_exchange_rate(from_curr, to_curr)
            converted = amount * rate
            return (f"Currency Conversion:\n"
                   f"Amount: {amount} {from_curr}\n"
                   f"Rate: 1 {from_curr} = {rate} {to_curr}\n"
                   f"Result: {converted:.2f} {to_curr}")
                   
        except Exception as e:
            logger.error(f"Exchange calculation error: {str(e)}")
            return "Couldn't perform currency conversion. Please check your request."

    async def calculate_balance_conversion(self, user: User, classification: Dict[str, Any]) -> str:
        """Convert account balance to another currency"""
        try:
            account = await Account.objects.aget(user=user)
            from_curr = classification.get("from_currency", "USD").upper()
            to_curr = classification.get("to_currency", "NPR").upper()
            
            rate = self._get_exchange_rate(from_curr, to_curr)
            converted = account.balance * rate
            return (f"Balance Conversion:\n"
                   f"Current Balance: {account.balance} {from_curr}\n"
                   f"Exchange Rate: 1 {from_curr} = {rate} {to_curr}\n"
                   f"Converted Balance: {converted:.2f} {to_curr}")
                   
        except Exception as e:
            logger.error(f"Balance conversion error: {str(e)}")
            return "Couldn't convert your balance. Please try again later."

    def _get_exchange_rate(self, from_curr: str, to_curr: str) -> Decimal:
        """Get exchange rate (mock implementation - replace with real API)"""
        rates = {
            "USD_NPR": Decimal("133.50"),
            "EUR_NPR": Decimal("145.25"),
            "GBP_NPR": Decimal("170.80"),
            "NPR_USD": Decimal("0.0075"),
            "NPR_EUR": Decimal("0.0069"),
            "NPR_GBP": Decimal("0.0059")
        }
        
        rate_key = f"{from_curr}_{to_curr}"
        if rate_key in rates:
            return rates[rate_key]
        
        # If direct rate not available, try via USD
        if f"{from_curr}_USD" in rates and f"USD_{to_curr}" in rates:
            return rates[f"{from_curr}_USD"] * rates[f"USD_{to_curr}"]
        
        raise ValueError(f"No exchange rate available for {from_curr} to {to_curr}")