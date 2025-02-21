import pandas as pd
from datetime import datetime
import ephem
import os

def calculate_moon_phase(date_str):
    """
    Calculate the moon phase for a given date.
    Returns a string describing the moon phase.
    """
    # Convert string to datetime
    date = datetime.strptime(date_str, '%m/%d/%Y')
    
    # Calculate moon phase using ephem
    moon = ephem.Moon()
    moon.compute(date)
    
    # Get illumination percentage
    illumination = moon.phase
    
    # Determine moon phase
    if illumination < 6.25:
        return 'New Moon'
    elif illumination < 43.75:
        return 'Waxing Crescent'
    elif illumination < 56.25:
        return 'First Quarter'
    elif illumination < 93.75:
        return 'Waxing Gibbous'
    elif illumination < 96.25:
        return 'Full Moon'
    elif illumination < 143.75:
        return 'Waning Gibbous'
    elif illumination < 156.25:
        return 'Last Quarter'
    else:
        return 'Waning Crescent'

# File path
file_path = 'playground/Lotto/lotteryLast30YrsJupiter.csv'

# Check if the file exists
if os.path.exists(file_path):
    print(f"File found: {file_path}")
    # Read the data
    data = pd.read_csv(file_path)
    
    # Check the first few rows of the data to ensure the date column is correct
    print(data.head())

    # Ensure the date column is in the correct format
    try:
        data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
    except Exception as e:
        print(f"Error converting date column: {e}")
        exit()

    # Add moon phase column
    data['moon_phase'] = data['date'].apply(lambda x: calculate_moon_phase(x.strftime('%m/%d/%Y')))

    # Display first few rows of the modified dataset
    print(data.head())

    # Save the modified dataset
    output_file_path = 'playground/Lotto/lottery_with_moon_phases.csv'
    data.to_csv(output_file_path, index=False)
    print(f"Modified dataset saved to: {output_file_path}")
else:
    print(f"File not found: {file_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in the current directory: {os.listdir()}")