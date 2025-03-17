from rest_framework import serializers
from .models import User, AccountType, Transactions, TransactionType

# class UserSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = User
#         fields = ['id', 'firstName', 'lastName', 'address', 'district', 'city', 'province',
#                   'dateOfBirth', 'panNumber', 'email', 'phone', 'username', 'accountNumber']
#         extra_kwargs = {'password': {'write_only': True}}

#     def create(self, validated_data):
#         request = self.context.get('request')
#         if request and not request.user.is_staff:
#             raise serializers.ValidationError("Only admins can create users.")
#         return super().create(validated_data)

# class AccountTypeSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = AccountType
#         fields = '__all__'

# class TransactionSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Transaction
#         fields = '__all__'

# class TransactionTypeSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = TransactionType
#         fields = '__all__'


# class ChatRequestSerializer(serializers.Serializer):
#     message = serializers.CharField(max_length=3024)

# class ChatResponseSerializer(serializers.Serializer):
#     response = serializers.CharField()

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