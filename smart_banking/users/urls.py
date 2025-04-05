from django.urls import path, include
from .views import  CreateUserView, CurrencyConversionView, LoginView, UserInfoView, CheckBalanceView,DepositView,WithdrawView,AccountStatementView, chat
#from .views import ChatbotView

# # Define the router for ViewSets
# router = DefaultRouter()
# router.register(r'users', UserViewSet)  # User model CRUD operations
# router.register(r'account-types', AccountTypeViewSet)  # Account types CRUD
# router.register(r'transactions', TransactionViewSet)  # Transactions CRUD
# router.register(r'transaction-types', TransactionTypeViewSet)  # Transaction types CRUD
#users/urls.py
urlpatterns = [
    path('users', CreateUserView.as_view(), name='create_user'),
    path('login', LoginView.as_view(), name='login'),
    path('me', UserInfoView.as_view(), name='user_info'),
    
    #path('api/', include(router.urls)),  # Include ViewSets for CRUD operations
    path('balance', CheckBalanceView.as_view(), name='check_balance'),  # Check balance (requires user auth)
    path('deposit', DepositView.as_view(), name='deposit_money'),  # Deposit money (requires user auth)
    path('withdraw', WithdrawView.as_view(), name='withdraw_money'),  # Withdraw money (requires user auth)
    path('account-statement', AccountStatementView.as_view(), name='view_transactions'),  # View transactions (requires user auth)
    path('chat', chat, name='chatbot_response'),
    path('convert-currency', CurrencyConversionView.as_view(), name='convert_currency'),
    #path("chat", ChatbotView.as_view(), name="chatbot"),
]


