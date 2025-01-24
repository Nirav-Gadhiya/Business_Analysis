import joblib
import pandas as pd
from datetime import datetime

# Load the saved model
model_file_path = 'random_forest_model_with_expense_salary.pkl'
loaded_model = joblib.load(model_file_path)

# Function to predict quantity based on user inputs with the option to skip (none are required)
def predict_quantity_from_user_input():
    print("\nEnter at least two details to predict the quantity sold. Press Enter to skip a field.")
    
    # Keep track of how many fields the user fills
    provided_inputs = 0
    expense = None
    employee_salary = None
    amount_earned = None
    date_obj = None
    
    # Ask for user input and ensure at least two fields are filled
    while provided_inputs < 2:
        # Get inputs (all are optional)
        if amount_earned is None:
            amount_earned_input = input("Amount Earned (Optional, press Enter to skip): ")
            if amount_earned_input:
                amount_earned = float(amount_earned_input)
                provided_inputs += 1
        if expense is None:
            expense_input = input("Expense (Optional, press Enter to skip): ")
            if expense_input:
                expense = float(expense_input)
                provided_inputs += 1
        if employee_salary is None:
            employee_salary_input = input("Employee Salary (Optional, press Enter to skip): ")
            if employee_salary_input:
                employee_salary = float(employee_salary_input)
                provided_inputs += 1
        if date_obj is None:
            date_input = input("Enter the Date (dd/mm/yyyy) (Optional, press Enter to skip): ")
            if date_input:
                date_obj = pd.to_datetime(date_input, format='%d/%m/%Y')
                provided_inputs += 1

        # If provided_inputs >= 2, break the loop
        if provided_inputs >= 2:
            break

        print("\nYou must provide at least two inputs. Please enter more data.")

    # If amount_earned was not provided, calculate it using a default formula
    if amount_earned is None:
        if expense is not None and employee_salary is not None:
            amount_earned = expense + (employee_salary * 2)  # Example assumption for calculating amount_earned

    # If the date is not provided, use the current date as a fallback
    if date_obj is None:
        date_obj = pd.to_datetime("today")

    # Calculate total_profit if not provided
    if expense is not None and amount_earned is not None:
        total_profit = amount_earned - expense
    else:
        total_profit = 0  # If expense or amount_earned is missing, set total_profit to 0

    # If total_profit is None, calculate it from the available parameters
    if total_profit is None and expense is not None and amount_earned is not None:
        total_profit = amount_earned - expense

    year, month, day = date_obj.year, date_obj.month, date_obj.day

    # Prepare the input data for prediction
    user_input = pd.DataFrame([[expense, employee_salary, amount_earned, total_profit, year, month, day]],
                              columns=["expense", "employee_salary", "amount_earned", "total_profit", "year", "month", "day"])

    # Scale the user input using the same scaler used for training data
    user_input_scaled = scaler.transform(user_input)

    # Predict the quantity sold
    predicted_quantity = model.predict(user_input_scaled)

    # Display the predicted quantity sold
    print(f"\nPredicted Quantity Sold: {predicted_quantity[0]:.2f}")

# Call the function to take user inputs and predict the quantity sold
predict_quantity_from_user_input()
