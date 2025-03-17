from django.contrib import admin
from .models import  Admin, AccountType, Loans, TransactionType, Account, Customers, User, Transactions

admin.site.register(Admin)
admin.site.register(AccountType)
admin.site.register(Loans)
admin.site.register(TransactionType)
admin.site.register(Account)
admin.site.register(Customers)
admin.site.register(User)
admin.site.register(Transactions)