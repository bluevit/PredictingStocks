import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 
import pyodbc
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Connect to the database
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=DESKTOP-N8MQKM6;"
    "Database=DB_STOCKMARKET;"
    "Trusted_Connection=yes"
)

for i in range(3):
    print('Connecting...')

print('\nConnected...')

initialquery = r"SELECT SYMBOL,MAX(DATE) AS LAST_UPDATE FROM TBL_STOCKPRICE GROUP BY SYMBOL;"
df01 = pd.read_sql_query(initialquery,conn)
print(df01)

query1 = r"SELECT * FROM VW_SUMMARY;"

#storing query result
df1 = pd.read_sql_query(query1,conn)
print(df1)

def pridection():
    symbol = input("Enter the symbol (AAPL,MSFT,NVDA): ")
    queryforsno = r"SELECT * FROM TBL_STOCKPRICE"
    dfsno = pd.read_sql_query(queryforsno, conn)
    dfsno['SNO'] = pd.to_numeric(dfsno['SNO'], errors='coerce')
    dfsno['SNO'] = dfsno['SNO'].astype(int)

    query = f"SELECT * FROM TBL_STOCKPRICE WHERE symbol = '{symbol}'"
    # Fetch the data into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Sort the DataFrame by date
    df['Date'] = pd.to_datetime(df['DATE'])
    df = df.sort_values(by='Date')

    # Feature selection
    features = ['OPENPRICE', 'HIGHPRICE', 'LOWPRICE', 'ADJCLOSE', 'VOL', 'CLOSEPRICE']
    target_features = ['OPENPRICE', 'HIGHPRICE', 'LOWPRICE', 'ADJCLOSE', 'VOL', 'CLOSEPRICE']

    # Create features and target variables
    X = df[features]
    y = df[target_features]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'\nMean Squared Error: {mse}')

    # Make predictions for the next day (based on the last date in the table)
    last_entry = df.iloc[-1][features].values.reshape(1, -1)
    last_entry_scaled = scaler.transform(last_entry)
    next_day_predictions = model.predict(last_entry_scaled)

    # Get the next day's date
    last_date = df['Date'].max()
    next_day_date = last_date + timedelta(days=1)

    # Create a DataFrame with the predicted values for the next day
    predicted_df = pd.DataFrame(next_day_predictions, columns=features)

    # Add the 'Symbol' and 'Date' columns to the predicted DataFrame
    predicted_df['Symbol'] = symbol
    predicted_df['Date'] = next_day_date

    next_day_df = pd.DataFrame()
    # Concatenate the predicted DataFrame with next_day_df
    next_day_df = pd.concat([next_day_df, predicted_df], ignore_index=True)

    # Display the DataFrame with predicted values for the next day
    next_day_df

    cursor = conn.cursor()
    # Insert Dataframe into SQL Server:
    for index, row in next_day_df.iterrows():
        cursor.execute("INSERT INTO PREDICTIONS(OPENPRICE,HIGHPRICE,LOWPRICE,ADJCLOSE,VOL,CLOSEPRICE,SYMBOL,DATE) values(?,?,?,?,?,?,?,?)",
                        row.OPENPRICE, row.HIGHPRICE, row.LOWPRICE,row.ADJCLOSE,row.VOL,row.CLOSEPRICE,row.Symbol,row.Date)
        
    
    last_sno = dfsno['SNO'].max()
    next_day_df['SNO'] = last_sno + 1

    reorder = ['SNO','Symbol','Date','OPENPRICE','HIGHPRICE','LOWPRICE','CLOSEPRICE','ADJCLOSE','VOL']
    next_day_df = next_day_df[reorder]

    for index, row in next_day_df.iterrows():
        cursor.execute("SET IDENTITY_INSERT TBL_STOCKPRICE ON INSERT INTO TBL_STOCKPRICE(SNO, SYMBOL, DATE, OPENPRICE, HIGHPRICE, LOWPRICE, CLOSEPRICE, ADJCLOSE, VOL) values(?,?,?,?,?,?,?,?,?)",
                       row.SNO, row.Symbol, row.Date, row.OPENPRICE, row.HIGHPRICE, row.LOWPRICE, row.CLOSEPRICE, row.ADJCLOSE, row.VOL)
    conn.commit()
    cursor.close()
    print("\nDatabase Updated...")

    querylast = f"SELECT * FROM TBL_STOCKPRICE WHERE symbol = '{symbol}'"
    # Fetch the data into a Pandas DataFrame
    df = pd.read_sql_query(querylast, conn)


    # Plotting the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(df['DATE'], df['CLOSEPRICE'], label='Historical Close Prices', color='blue')

    # Mark the last close price with a different color
    last_close_price = df['CLOSEPRICE'].iloc[-1]
    last_date = df['DATE'].iloc[-1]
    plt.scatter(last_date, last_close_price, color='red', marker='o', label=f'Last Close Price ({last_close_price:.2f})')

    # Add title and labels
    plt.title(f'Historical Close Prices and Predicted Close Price for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def seasonal_analysis():
    symbol = input("Enter the symbol (AAPL,MSFT,NVDA): ")
    sea_query = f"SELECT MONTH(DATE) AS Month, AVG(CLOSEPRICE) AS AvgClosePrice FROM TBL_STOCKPRICE WHERE SYMBOL = '{symbol}' GROUP BY MONTH(DATE) ORDER BY Month;"
    # Fetch the data into a Pandas DataFrame
    sea_df = pd.read_sql_query(sea_query, conn)
    print(sea_df)


def trading_vol_analysis():
    symbol = input("Enter the symbol (AAPL,MSFT,NVDA): ")
    vol_query = f"SELECT DATE, VOL FROM TBL_STOCKPRICE WHERE SYMBOL = '{symbol}' ORDER BY DATE;"
    # Fetch the data into a Pandas DataFrame
    vol_df = pd.read_sql_query(vol_query, conn)
    print(vol_df)

columns = ["Choice", "Action"]

myTable = PrettyTable()
myTable.add_column(columns[0], ["1", "2", "3","4"])
myTable.add_column(columns[1], ["Seasonal Analysis", "Trading Volume Analysis", "Predict for Next Day" ,"Exit"])
#myTable.title = "Welcome to DS Tool"
print(myTable)
print("Expecting Input...")

while True:
    choice=input("Enter a choice 1-4\n")
    match choice:
        case "1":
            seasonal_analysis()
        case "2":
            trading_vol_analysis()
        case "3":
            pridection()
    if choice == "4":
        conn.close()
        print("\nKeep Analyzing!!!")
        break
