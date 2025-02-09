import serial
import csv
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Model preparation
df = pd.read_csv('./Orange Quality Data.csv')
df = df.drop(['Size (cm)', 'Weight (g)', 'HarvestTime (days)', 'Color', 'Variety', 'Blemishes (Y/N)'], axis=1)
X = df.drop('Quality (1-5)', axis=1)
y = df['Quality (1-5)']

# Train test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(scaled_X_train, y_train)
SERIAL_PORT = "COM7"
BAUD_RATE = 115200
OUTPUT_FILE = "sensor_data.csv"
UPDATE_INTERVAL = 4  

def predict_quality(data_row):
    features = np.array([[
        data_row['Ripeness'],
        data_row['pH'],
        data_row['Brix'],
        data_row['Softness']
    ]])
    scaled_features = scaler.transform(features)
    prediction = ridge_model.predict(scaled_features)[0]
    return round(prediction * 2) / 2

def main():
   
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  
    
    print("Starting orange quality monitoring...")
    print("Press Ctrl+C to stop.")
    
    last_write_time = 0
    
    try:
        while True:
            current_time = time.time()
            
            if current_time - last_write_time >= UPDATE_INTERVAL:
                with open(OUTPUT_FILE, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Ripeness", "pH", "Brix", "Softness"])
                    
                    
                    line = ser.readline().decode("utf-8").strip()
                    if line:
                        data = line.split(",")
                        if len(data) == 4:
                            ripeness = int(data[0])
                            pH = float(data[1])
                            brix = float(data[2])
                            softness = int(data[3])
                            
                            writer.writerow([ripeness, pH, brix, softness])
                            
                            data_row = {
                                'Ripeness': ripeness,
                                'pH': pH,
                                'Brix': brix,
                                'Softness': softness
                            }
                            
                            
                            quality_prediction = predict_quality(data_row)
                            
                            
                            print("\n" + "="*50)
                            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"Sensor Readings:")
                            print(f"  Ripeness: {ripeness}")
                            print(f"  pH: {pH:.2f}")
                            print(f"  Brix: {brix:.2f}")
                            print(f"  Softness: {softness}")
                            print(f"Predicted Quality: {quality_prediction:.1f}/5.0")
                            print("="*50)
                            
                last_write_time = current_time
            
            
            time.sleep(0.1)
                    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        ser.close()
        print("Serial connection closed.")

if __name__ == "__main__":
    main()