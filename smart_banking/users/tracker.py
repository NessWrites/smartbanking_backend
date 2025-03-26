import pandas as pd
import joblib
from django.db import models
import os
from django.conf import settings
import requests

from users.config import API_KEY_DISTANCE_AI, API_KEY_TWILIO, ACCOUNT_SID_TWILIO, VERIFY_SERVICE_SID_TWILIO
from twilio.rest import Client

class AccountNumberTracker(models.Model):
    id = models.AutoField(primary_key=True)
    last_account_number = models.PositiveIntegerField(default=100000)
    
    def __str__(self):
        return f"Tracker {self.id} - Last Account Number: {self.last_account_number}"

class Tracker:
    def __init__(self):
        # Initialize Twilio client
        self.twilio_client = Client(ACCOUNT_SID_TWILIO, API_KEY_TWILIO)
        self.verify_service_sid = VERIFY_SERVICE_SID_TWILIO
        
        # Load the XGBoost model
        model_path = "/Users/ness/Islington/FYP_smart_banking/backend/smart_banking/users/xgboost_fraud_model.pkl"
        self.boost = joblib.load(model_path)

    def distance_from_home(self, db_home, atm_location):
        """
        Calculate the distance between the user's home and the ATM location.
        """
        origin = db_home
        destination = atm_location
        print(origin)
        print(destination)
        url = f'https://api.distancematrix.ai/maps/api/distancematrix/json?origins={origin}&destinations={destination}&key={API_KEY_DISTANCE_AI}'
        response = requests.get(url,timeout=10)
        data = response.json()

        if data['status'] == 'OK':
            element = data['rows'][0]['elements'][0]
            if element['status'] == 'OK':
                distance = element['distance']['text']
                miles = float(distance.split()[0]) * 0.621371
                return miles
            else:
                raise Exception(f'Error in element: {element["status"]}')
        else:
            raise Exception(f'Error in response: {data["status"]}')

    def send_otp(self, phone_number):
        """
        Send an OTP to the user's phone number using Twilio.
        """
        phone_number = f'+977{phone_number}'
        verification = self.twilio_client.verify.v2.services(self.verify_service_sid) \
            .verifications.create(to=phone_number, channel="sms")
        return verification.status

    def verify_otp(self, phone_number, otp_code):
        phone_number = f'+977{phone_number}'
        """
        Verify the OTP entered by the user.
        """
        verification_check = self.twilio_client.verify.v2.services(self.verify_service_sid) \
            .verification_checks.create(to=phone_number, code=otp_code)
        return verification_check.status == "approved"
    
    def check_for_fraud(self, user, amount,last_location, location,repeat_retailer, withdrawal_type,ratio_to_median_purchase_price):
        """
        Check if the transaction seems fraudulent using the XGBoost model.
        """
        user_address = f"{user.address},  {user.district}, {user.provinces}"

    
        # Calculate distance from home
        distance_from_home = self.distance_from_home(user_address, location)
    
        # Calculate distance from last transaction
        distance_from_last_transaction = self.distance_from_home(last_location, location)
    
        
        
        # Prepare the input data for the XGBoost model
        input_data = pd.DataFrame({
            'distance_from_home': [distance_from_home],  # Example feature
            'distance_from_last_transaction': [distance_from_last_transaction],  # Example feature
            'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],  # Example feature
            'repeat_retailer': [repeat_retailer],  # Example feature
            'used_chip': [1 if withdrawal_type == "online_order" else 0],  # Example feature
            'used_pin_number': [0],  # Example feature
            'online_order': [1 if withdrawal_type == "online_order" else 0]  # Example feature
        })

        # Preprocess the input data
        input_data = self.preprocess(input_data)
        # Debug
        print("Prepared DataFrame:\n", input_data)
        print("\nData types:\n", input_data.dtypes)

        # Predict fraud probability using the XGBoost model
        fraud_probability = self.boost.predict_proba(input_data)[:, 1]

        # Check if the fraud probability exceeds the threshold
        if fraud_probability >= 0.578947:  # Threshold determined earlier
            return True
        else:
            return False

    def preprocess(self, data):
        """
        Preprocess the input data for the XGBoost model.
        """
        # Convert data types
        data[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']] = data[['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']].astype('int')
        return data