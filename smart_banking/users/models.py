# Create  models here.
from decimal import Decimal
from django.conf import settings
from django.db import models
from .tracker import AccountNumberTracker
from django.dispatch import receiver
from django.db.models.signals import post_save
from smart_banking.settings import AUTH_USER_MODEL
from django.contrib.auth.models import AbstractBaseUser,User, BaseUserManager, PermissionsMixin
from django.db import models
import statistics
from datetime import date, timedelta
from decimal import Decimal
from django.utils import timezone
import requests  # Add this line at the top of models.py



# Custom User Manager
class UserManager(BaseUserManager):
    def create_user(self, phoneNumber, password=None, **extra_fields):
        if not phoneNumber:
            raise ValueError("Phone number is required")
        user = self.model(phoneNumber=phoneNumber, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, phoneNumber, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")
        
        return self.create_user(phoneNumber, password, **extra_fields)

#2 Admin Model
class Admin(models.Model):
    adminID = models.AutoField(primary_key=True)
    department = models.CharField(max_length=100)
    salary = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"Admin {self.adminID} - {self.department}"

#4 Loans Model
class Loans(models.Model):
    loanID = models.AutoField(primary_key=True)
    loanType = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    interestRate = models.DecimalField(max_digits=5, decimal_places=2)
    minAmount = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    maxAmount = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    minTerm = models.IntegerField(blank=True, null=True)
    maxTerm = models.IntegerField(blank=True, null=True)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True, null=True)  # Allow NULL values

    def __str__(self):
        return f"{self.loanType} ({self.interestRate}%)"



#3 AccountType Model
class AccountType(models.Model):
    accountTypeID = models.AutoField(primary_key=True)
    depositType = models.CharField(max_length=100)
    depositRates = models.DecimalField(max_digits=5, decimal_places=2)
    #userID = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.depositType} ({self.depositRates}%)"

# Account Model
class Account(models.Model):
    accountNumber = models.BigIntegerField(primary_key=True, unique=True)
    accountTypeID = models.ForeignKey(AccountType, on_delete=models.CASCADE)
    balance = models.DecimalField(max_digits=15, decimal_places=2, default=0.00)
    loanID = models.ForeignKey(Loans, on_delete=models.SET_NULL, null=True, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    def __str__(self):
        return f"Account {self.accountNumber}"


#1 Users Model
class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    phoneNumber = models.CharField(max_length=15,unique=True, null=False, blank=False)
    firstName = models.CharField(max_length=100)
    lastName = models.CharField(max_length=100)
    address = models.CharField(max_length=50, null = False, blank=False)
    district = models.CharField(max_length=50, null = False, blank=False)
    city = models.CharField(max_length=50, null = False, blank=False)
    provinces = models.CharField(max_length=50, null = False, blank=False)
    dateOfBirth = models.DateField(null=True, blank=True)
    panNumber = models.CharField(max_length=20, unique=True)
    createdAt = models.DateTimeField(auto_now_add=True)
    accountNumber = models.PositiveIntegerField(unique=True, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    USERNAME_FIELD = "phoneNumber"  # Login with email
    REQUIRED_FIELDS = ["firstName"]
    objects = UserManager()
    
    
    def save(self, *args,  **kwargs):
        if not self.accountNumber:
            self.accountNumber = self.generate_account_number()
        super(User, self).save(*args, **kwargs)
        
        # Create an Account for the User if it doesn't already exist
        if not Account.objects.filter(user=self).exists():  # Check if the User already has an Account
            try:
                # Fetch the "Saving Account" type (id=2)
                account_type = AccountType.objects.get(accountTypeID=2)
                print(f"Fetched AccountType: {account_type}")  # Debugging
                Account.objects.create(
                    accountNumber=self.accountNumber,  # Use the User's accountNumber
                    accountTypeID=account_type,
                    balance=self.account_balance,  # Use the User's default balance
                    user=self  # Link the Account to the User
                )
            except AccountType.DoesNotExist:
                    print("Error: AccountType with id=2 does not exist.")  # Debugging


    def generate_account_number(self):
        
        tracker, created = AccountNumberTracker.objects.get_or_create(id=1)
        tracker.last_account_number += 1
        tracker.save()
        return tracker.last_account_number
    @property
    def account_balance(self):
        account = Account.objects.filter(user=self).first()
        if account:
            return account.balance
        return Decimal('0.00')  # Default if no account yet
    
    
    def get_transaction_details(self):
        # Get all transactions related to the user
        transactions = Transactions.objects.filter(accountID__user=self)
        return [
            {
                "transactionID": trans.transactionID,
                "transactionType": trans.transactionTypeID.transactionType,
                "amount": trans.amount,
                "reference": trans.reference,
                "description": trans.description,
                "date": trans.date,
            }
            for trans in transactions
        ]

    def get_withdraw_details(self):
        # Get all withdrawals related to the user
        withdraws = Withdraw.objects.filter(user=self)
        return [
            {
                "withdrawID": withdraw.withdrawID,
                "withdrawalType": withdraw.withdrawal_type,
                "amount": withdraw.transaction.amount,
                "home": withdraw.home,
                "location": withdraw.location,
                "retailer": withdraw.retailer,
                "created_at": withdraw.created_at,
            }
            for withdraw in withdraws
        ]

    def get_user_data(self):
        # Fetch account balance
        account = Account.objects.filter(user=self).first()
        balance = account.balance if account else Decimal('0.00')

        # Fetch transaction and withdrawal details
        transaction_details = self.get_transaction_details()
        withdraw_details = self.get_withdraw_details()

        # Return a dictionary with all placeholders
        return {
            
    #Customer ID
    "Customer ID":self.id,        
    # First Name variations
    "firstName": self.firstName,
    "first_name": self.firstName,
    "first Name": self.firstName,
    "Account Name": self.firstName + self.lastName,
    "Customer Name": self.firstName + self.lastName,

    # Last Name variations
    "lastName": self.lastName,
    "last_name": self.lastName,
    "last Name": self.lastName,

    # Phone Number variations
    "phoneNumber": self.phoneNumber,
    "phone_number": self.phoneNumber,
    "phone Number": self.phoneNumber,

    # Email variations
    "email": self.email,
    "e-mail": self.email,

    # Account Number variations
    "accountNumber": self.accountNumber,
    "account_number": self.accountNumber,
    "account Number": self.accountNumber,
    "Account Number": self.accountNumber,

    # Address variations
    "address": self.address,

    # District variations
    "district": self.district,

    # City
    "city": self.city,

    # Provinces
    "provinces": self.provinces,

    # Date of Birth variations
    "dateOfBirth": self.dateOfBirth,
    "date_of_birth": self.dateOfBirth,
    "date of birth": self.dateOfBirth,

    # PAN Number variations
    "panNumber": self.panNumber,
    "pan_number": self.panNumber,
    "pan number": self.panNumber,

    # Balance / Account Balance
    "account_balance": balance,
    "Account Balance": balance,
    "account balance": balance,
    "Balance": balance,
    "check balance":balance,
    "Check Balance":balance,
    "balance": balance,

    # Created At variations
    "createdAt": self.createdAt,
    "created_at": self.createdAt,
    "created at": self.createdAt,
    
    #transaction history
    "transaction_history": self.get_transaction_details(),
    "withdraw_history": self.get_withdraw_details(),

    # Transaction History variations
    "transaction_history": transaction_details,
    "transaction history": transaction_details,

    # Withdraw History variations
    "withdraw_history": withdraw_details,
    "withdraw history": withdraw_details,

    # Customer Support Info
    "customer_support_working_hours": "9 AM - 6 PM",
    "customer support working hours": "9 AM - 6 PM",
    "customer_support_phone_number": "+977-9841467002",
    "customer support phone number": "+977-9841467002",
    "Customer Support Working Hours": "9 AM - 6 PM",

    # Telephone Banking Number variations
    "telephone_banking_number": "+977-9841467002",
    "telephone banking number": "+977-9841467002",
    "Customer Service Phone Number": "+977-9841467002", 
    

    # Website variations
    "company_website": "https://nischhalshrestha.com",
    "company website": "https://nischhalshrestha.com",
    "company_website_url": "https://nischhalshrestha.com",
    "company website url": "https://nischhalshrestha.com",
    "bank_website": "https://nischhalshrestha.com",
    "bank website": "https://nischhalshrestha.com",
    "bank_website_url": "https://nischhalshrestha.com",
    "bank website url": "https://nischhalshrestha.com",
    "Bank's Online Banking Portal":"https://nischhalshrestha.com",
    "Company Website URL": "https://nischhalshrestha.com", 
    "Banking App": "Smart Banking",
    "Bank Name": "Smart Banking",
    "Banking app name": "Smart Banking",
    "bank name": "Smart Banking",
    "Customer Support": "Smart Banking",
}

    
    def median_purchase_price(self):
        # Fetch all transactions for the user
        transactions = Transactions.objects.filter(accountID__user=self)
        amounts = [t.amount for t in transactions]
        
        # Calculate the median
        if amounts:
            return Decimal(statistics.median(amounts))
        else:
            return Decimal('0.00')  # Default value if no transactions exist



    def __str__(self):
        return f"{self.id} - {self.firstName} {self.lastName} ({self.email})"





#4 TransactionType Model
class TransactionType(models.Model):
    TRANSACTION_CHOICES = [
        ("withdraw", "Withdraw"),
        ("deposit", "Deposit"),
        ("transfer", "Transfer"),
    ]
    transactionTypeID = models.AutoField(primary_key=True)
    transactionType = models.CharField(max_length=10, choices=TRANSACTION_CHOICES, unique=True)
    #userID = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.transactionType

#5 Transactions Model
class Transactions(models.Model):
    transactionID = models.AutoField(primary_key=True)
    date = models.DateTimeField(auto_now_add=True)
    reference = models.CharField(max_length=255)
    description = models.TextField()
    transactionTypeID = models.ForeignKey(TransactionType, on_delete=models.CASCADE)
    accountID = models.ForeignKey('Account', on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=15, decimal_places=2, null = True)

    def __str__(self):
        return f"Transaction {self.transactionID} - {self.reference}"


#7 Customers Model
class Customers(models.Model):
    customerID = models.AutoField(primary_key=True)
    beneficiary = models.CharField(max_length=255)
    #accountID = models.ForeignKey(Account, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.customerID} - {self.beneficiary}"
    
class Withdraw(models.Model):
    WITHDRAWAL_TYPE_CHOICES = [
        ('atm', 'ATM Withdrawal'),
        ('online_order', 'Online Order'),
    ]

    withdrawID = models.AutoField(primary_key=True)
    transaction = models.OneToOneField(Transactions, on_delete=models.CASCADE, related_name='withdraw')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='withdrawals')
    home = models.CharField(max_length=255)  # Auto-filled from User model
    location = models.CharField(max_length=255)  # Branch location from dropdown
    last_location = models.CharField(max_length=255)  # Auto-filled from previous withdrawal
    retailer = models.CharField(max_length=255, null=True, blank=True)  # For online orders only
    repeat_retailer = models.BooleanField(default=False)  # Auto-calculated in save()
    used_chip = models.BooleanField(default=False)  # Optional for ATM
    used_pin = models.BooleanField(default=True)  # Optional for ATM
    online_order = models.BooleanField(default=False)  # Optional for ATM
    withdrawal_type = models.CharField(max_length=20, choices=WITHDRAWAL_TYPE_CHOICES)  # 'atm' or 'online_order'
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # Auto-assign user's address to 'home' if not set
        if not self.home:
            self.home = f"{self.user.address}, {self.user.city}, {self.user.district}, {self.user.provinces}"

        # Get last location
        last_withdrawal = Withdraw.objects.filter(user=self.user).exclude(pk=self.pk).order_by('-created_at').first()
        self.last_location = last_withdrawal.location if last_withdrawal else self.location

        # Set repeat_retailer if online_order
        if self.withdrawal_type == 'online_order' and self.retailer:
            self.repeat_retailer = Withdraw.objects.filter(
                user=self.user,
                retailer=self.retailer,
                withdrawal_type='online_order'
            ).exists()

        super(Withdraw, self).save(*args, **kwargs)
        
    

    def __str__(self):
        return f"Withdraw {self.withdrawID} - {self.withdrawal_type} - {self.transaction.reference}"

class CurrencyExchange:
    """
    Handles currency exchange operations using NRB API.
    """
    BASE_URL = "https://www.nrb.org.np/api/forex/v1/rates"
    
    @classmethod
    def get_exchange_rate(cls, date, currency):
        """
        Fetches exchange rate for the given date and currency from NRB API.
        Returns buy/sell rates for the currency against NPR.
        """
        params = {
            "from": date,
            "to": date,
            "per_page": 50,
            "page": 1
        }

        response = requests.get(cls.BASE_URL, params=params)
        data = response.json()

        if data["status"]["code"] == 200:
            for rate_info in data["data"]["payload"]:
                for rate in rate_info["rates"]:
                    if rate["currency"]["iso3"] == currency:
                        return {
                            "buy": float(rate["buy"]),   # NPR per unit of foreign currency
                            "sell": float(rate["sell"])  # NPR per unit of foreign currency
                        }
        return None

    @classmethod
    def convert_currency(cls, amount, date, from_currency, to_currency):
        """
        Returns both converted amount and exchange rate
        """
        if from_currency == to_currency:
            return {
                "converted_amount": amount,
                "exchange_rate": 1,
                "success": True
            }
        
        # Get rates for both currencies
        from_rate = cls.get_exchange_rate(date, from_currency) if from_currency != "NPR" else None
        to_rate = cls.get_exchange_rate(date, to_currency) if to_currency != "NPR" else None
    
        # Conversion logic
        if from_currency == "NPR" and to_rate:
            exchange_rate = 1 / to_rate["sell"]
            return {
                "converted_amount": amount * exchange_rate,
                "exchange_rate": exchange_rate,
                "success": True
            }
        elif to_currency == "NPR" and from_rate:
            exchange_rate = from_rate["buy"]
            return {
                "converted_amount": amount * exchange_rate,
                "exchange_rate": exchange_rate,
                "success": True
            }
        elif from_rate and to_rate:
            exchange_rate = from_rate["buy"] / to_rate["sell"]
            return {
                "converted_amount": amount * exchange_rate,
                "exchange_rate": exchange_rate,
                "success": True
            }
        
        return {"success": False, "error": "Exchange rate not available"}