import gradio as gr
import yfinance as yf
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
import requests

def train_and_predict(stock_name, lookback=60, days_to_predict=10):
    # Fetch stock data
    data = yf.download(stock_name, period="1y")
    closing_prices = data['Close'].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)

    # Prepare training data
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    # Predict the next 10 days' stock prices
    predictions = []
    last_sequence = scaled_data[-lookback:]  # Last `lookback` days as the starting point
    
    for _ in range(days_to_predict):
        # Reshape last sequence for prediction
        last_sequence_reshaped = last_sequence.reshape((1, lookback, 1))
        predicted_price_scaled = model.predict(last_sequence_reshaped)
        
        # Convert the scaled prediction back to original price
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        predictions.append(float(predicted_price[0][0]))
        
        # Update last_sequence by appending the new prediction and removing the oldest entry
        last_sequence = np.append(last_sequence, predicted_price_scaled, axis=0)
        last_sequence = last_sequence[1:]  # Keep only the last `lookback` elements

    return predictions


def get_snippets(stock_name):
    stock_name = stock_name.replace('.NS', '')
    url = f"https://munafasutra.com/nse/tomorrow/{stock_name}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Adjust selectors based on the page structure to extract the relevant data
        data = soup.find_all('div', id='munafaValue')  # Replace with the correct HTML element
        return [item.get_text() for item in data]
    else:
        return "Failed to retrieve data"
    
# Gradio interface
with gr.Blocks() as demo:
    stock_name_input = gr.Textbox(label="Enter Stock Symbol", placeholder="e.g., AAPL")
    lookback_input = gr.Slider(minimum=10, maximum=100, value=60, step=1, label="Lookback Days")
    predict_button = gr.Button("Train Model & Predict Next 10 Days Price")
    prediction_output = gr.Textbox(label="Predicted Stock Prices for Next 10 Days", interactive=False, lines=10)

    predict_button.click(
        fn=lambda stock, lookback: "\n".join([f"Day {i+1}: {price:.2f}" for i, price in enumerate(train_and_predict(stock, lookback))]),
        inputs=[stock_name_input, lookback_input],
        outputs=prediction_output
    )

    # Section for fetching snippets
    gr.Markdown("# Stock Opinion Snippets")
    output_textbox = gr.Textbox(label="Snippets", interactive=False, lines=10)

    submit_btn = gr.Button("Get Snippets")
    submit_btn.click(get_snippets, inputs=stock_name_input, outputs=output_textbox)

# Launch the Gradio app
demo.launch()