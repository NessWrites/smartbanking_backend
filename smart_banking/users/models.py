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
    loanTypes = models.CharField(max_length=100)
    interestRates = models.DecimalField(max_digits=5, decimal_places=2)
    #userID = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.loanTypes} ({self.interestRates}%)"


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
    # addressID = models.ForeignKey(Address, on_delete=models.CASCADE)
    # adminID = models.ForeignKey(Admin, on_delete=models.SET_NULL, null=True, blank=True)
    # customerID = models.ForeignKey(Customers, on_delete=models.SET_NULL, null=True, blank=True)
    address = models.CharField(max_length=50, null = False, blank=False)
    district = models.CharField(max_length=50, null = False, blank=False)
    city = models.CharField(max_length=50, null = False, blank=False)
    provinces = models.CharField(max_length=50, null = False, blank=False)
    dateOfBirth = models.DateField(null=True, blank=True)
    panNumber = models.CharField(max_length=20, unique=True)
    createdAt = models.DateTimeField(auto_now_add=True)
    accountNumber = models.PositiveIntegerField(unique=True, blank=True, null=True)
    #account_balance = models.DecimalField(max_digits=15, decimal_places=2, default=500.00)  # Add balance field
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
