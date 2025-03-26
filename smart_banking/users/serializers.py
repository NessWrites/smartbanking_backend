from rest_framework import serializers
from .models import (
     Admin, AccountType, Loans, TransactionType,
    Account, Customers, User, Transactions
)
# Users Serializer
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['firstName', 'lastName',
                  'address', 'district', 'city',
                  'provinces', 'dateOfBirth', 'panNumber',
                  'email', 'phoneNumber', 'password', 'accountNumber','account_balance']
        extra_kwargs = {'password': {'write_only': True}}


    def create(self, validated_data):
        password = validated_data.pop('password')  # Extract password
        user = User(**validated_data)  # Create user instance without password
        user.set_password(password)  # Hash the password
        user.save()  # Save the user
        return user





# Admin Serializer
class AdminSerializer(serializers.ModelSerializer):
    class Meta:
        model = Admin
        fields = '__all__'

# AccountType Serializer
class AccountTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = AccountType
        fields = '__all__'

# Loans Serializer
class LoansSerializer(serializers.ModelSerializer):
    class Meta:
        model = Loans
        fields = '__all__'

# TransactionType Serializer
class TransactionTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = TransactionType
        fields = '__all__'

# Account Serializer
class AccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = Account
        fields = '__all__'

# Customers Serializer
class CustomersSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customers
        fields = '__all__'

# Transactions Serializer
class TransactionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transactions
        fields = '__all__'