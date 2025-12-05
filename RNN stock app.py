import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import random

# ---------------------------------------------------------
# 설정 (Configuration)
# ---------------------------------------------------------
CONFIG = {
    'start_date': '2021-01-01',
    'end_date': '2025-11-30',
    'seq_length': 10,
    'input_dim': 6,
    'hidden_dim': 64,
    'output_dim': 1,
    'num_layers': 3,
    'epochs': 50,  # 에포크 수를 줄여서 빠른 테스트
    'learning_rate': 0.001,
    'batch_size': 64,
    'static_input_dim': 3,
    'static_hidden_dim': 32,
    'test_split_ratio': 0.2
}

# ---------------------------------------------------------
# 1. 데이터 처리 클래스 (Data Processing)
# ---------------------------------------------------------
class StockDataProcessor:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
    
    def get_sp500_it_tickers(self):
        """
        Reads S&P 500 IT sector tickers from Wikipedia.
        """
        try:
            print("Fetching S&P 500 IT sector tickers from Wikipedia...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0'}
            tables = pd.read_html(url, storage_options={'User-Agent': headers})
            sp500_df = tables[0]
            
            it_tickers = sp500_df[sp500_df['GICS Sector'] == 'Information Technology']['Symbol'].tolist()
            # Some symbols in Wikipedia might have '.' replaced with '-', yfinance needs '.'
            it_tickers = [ticker.replace('.', '-') for ticker in it_tickers]
            
            print(f"Found {len(it_tickers)} tickers in the IT sector.")
            return it_tickers
        except Exception as e:
            print(f"Could not fetch tickers: {e}")
            # Fallback to a small list if fetching fails
            return ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'ACN', 'ORCL', 'ADBE', 'CRM', 'INTC', 'CSCO']

    def get_and_split_tickers(self, test_size=0.2):
        """
        Gets the ticker list and splits it into training and testing sets.
        """
        tickers = self.get_sp500_it_tickers()
        if not tickers:
            raise ValueError("Ticker list is empty.")
            
        train_tickers, test_tickers = train_test_split(tickers, test_size=test_size, random_state=42)
        print(f"Total {len(tickers)} tickers split into {len(train_tickers)} training and {len(test_tickers)} testing tickers.")
        return train_tickers, test_tickers

    def build_dataset(self, tickers, seq_length):
        """
        Builds a combined dataset for a given list of tickers.
        """
        all_x_seq, all_y, all_x_static = [], [], []

        for i, ticker in enumerate(tickers):
            print(f"\nProcessing {i+1}/{len(tickers)}: {ticker}")
            
            try:
                # 1. Download price data
                df = yf.download(ticker, start=self.start, end=self.end, progress=False)
                if df.empty:
                    print(f"No data for {ticker}, skipping.")
                    continue

                # 2. Get fundamental data
                ticker_info = yf.Ticker(ticker).info
                per = ticker_info.get('trailingPE', 0)
                pbr = ticker_info.get('priceToBook', 0)
                roe = ticker_info.get('returnOnEquity', 0)
                static_data = np.array([per if per else 0, pbr if pbr else 0, roe if roe else 0], dtype=np.float32)

                # 3. Feature Engineering
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df = df.dropna()

                features_df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5']]
                
                # 4. Scale and Create Sequences
                x_values = self.scaler_x.fit_transform(features_df.values)
                y_values = self.scaler_y.fit_transform(features_df[['Close']].values)
                
                if len(x_values) < seq_length:
                    print(f"Not enough data for {ticker} to create a sequence, skipping.")
                    continue
                    
                xs, ys, static_list = self._create_sequences_for_ticker(x_values, y_values, seq_length, static_data)

                all_x_seq.append(xs)
                all_y.append(ys)
                all_x_static.append(static_list)

            except Exception as e:
                print(f"An error occurred while processing {ticker}: {e}")

        # Concatenate all data from all tickers
        final_x_seq = np.concatenate(all_x_seq, axis=0)
        final_y = np.concatenate(all_y, axis=0)
        final_x_static = np.concatenate(all_x_static, axis=0)
        
        return final_x_seq, final_y, final_x_static

    def _create_sequences_for_ticker(self, x_data, y_data, seq_length, static_data):
        xs, ys, static_list = [], [], []
        for i in range(len(x_data) - seq_length):
            x_window = x_data[i : i+seq_length]
            y_label = y_data[i + seq_length]
            
            xs.append(x_window)
            ys.append(y_label)
            static_list.append(static_data)
            
        return np.array(xs), np.array(ys), np.array(static_list)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)

# ---------------------------------------------------------
# 2. 모델 정의 (Stacked RNN)
# ---------------------------------------------------------
class HybridRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, static_input_dim, static_hidden_dim):
        super(HybridRNN, self).__init__()
        
        # 1. 시계열 데이터 처리를 위한 RNN 층
        self.rnn = nn.RNN(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 2. 고정 데이터 처리를 위한 Linear 층
        self.static_layer = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU()
        )
        
        # 3. 결합된 데이터를 처리할 최종 출력층
        self.fc = nn.Linear(hidden_dim + static_hidden_dim, output_dim)

    def forward(self, x_seq, x_static):
        # 1. 시계열 데이터 처리
        out, h_n = self.rnn(x_seq)
        rnn_last_hidden = h_n[-1] # (Batch, Hidden)
        
        # 2. 고정 데이터 처리
        static_out = self.static_layer(x_static) # (Batch, StaticHidden)
        
        # 3. 데이터 결합 (Concatenate)
        combined = torch.cat((rnn_last_hidden, static_out), dim=1)
        
        # 4. 최종 예측
        output = self.fc(combined)
        return output

# ---------------------------------------------------------
# 3. 학습 및 평가 함수
# ---------------------------------------------------------
def train_model(model, X_train_seq, X_train_static, y_train, config):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    dataset = TensorDataset(X_train_seq, X_train_static, y_train)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    model.train()
    print(f"\nStarting Training (Hybrid RNN) with {len(data_loader.dataset)} samples...")
    
    loss_history = []
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for seq_batch, static_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = model(seq_batch, static_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        loss_history.append(avg_epoch_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_epoch_loss:.6f}")
            
    return loss_history

def evaluate_model(model, X_test_seq, X_test_static, y_test, processor):
    model.eval()
    with torch.no_grad():
        predicted = model(X_test_seq, X_test_static)
    
    predicted_price = processor.scaler_y.inverse_transform(predicted.numpy())
    real_price = processor.scaler_y.inverse_transform(y_test.numpy())
    
    rmse = math.sqrt(mean_squared_error(real_price, predicted_price))
    return predicted_price, real_price, rmse

# ---------------------------------------------------------
# 4. 결과 로깅 함수
# ---------------------------------------------------------
def log_results(log_file, timestamp, model_name, training_time, final_loss, rmse):
    log_entry = (
        f"Timestamp: {timestamp}, "
        f"Model: {model_name}, "
        f"TrainingTime: {training_time:.2f}s, "
        f"FinalLoss: {final_loss:.6f}, "
        f"TestRMSE: {rmse:.4f}\n"
    )
    try:
        with open(log_file, 'a') as f:
            f.write(log_entry)
        print(f"\nResults successfully logged to {log_file}")
    except IOError as e:
        print(f"Error logging results: {e}")

def validate_and_plot_ticker(model, ticker, processor, config, timestamp):
    """
    Validates the model on a single ticker and saves a plot with offset correction.
    """
    print(f"\n--- Validating on individual ticker: {ticker} ---")
    
    # 1. Build dataset for the single ticker
    X_seq_np, y_np, X_static_np = processor.build_dataset([ticker], config['seq_length'])
    
    if X_seq_np.shape[0] == 0:
        print(f"Could not create dataset for {ticker}. Skipping validation.")
        return

    # 2. Convert to tensors
    X_seq = torch.tensor(X_seq_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    X_static = torch.tensor(X_static_np, dtype=torch.float32)

    # 3. Evaluate
    predicted_price, real_price, original_rmse = evaluate_model(model, X_seq, X_static, y, processor)
    print(f"Original RMSE for {ticker}: {original_rmse:.4f}")

    # 4. [추가] 예측값 보정 (Offset Correction)
    # 실제값의 첫 번째 지점과 예측값의 첫 번째 지점의 차이를 offset으로 사용
    if len(real_price) > 0 and len(predicted_price) > 0:
        offset = real_price[0] - predicted_price[0]
        adjusted_predicted_price = predicted_price + offset
        corrected_rmse = math.sqrt(mean_squared_error(real_price, adjusted_predicted_price))
        print(f"Corrected RMSE for {ticker}: {corrected_rmse:.4f}")
    else:
        adjusted_predicted_price = predicted_price
        corrected_rmse = original_rmse
        print(f"Not enough data for offset correction. Using original RMSE.")

    # 5. Plot and save
    plt.figure(figsize=(12, 6))
    plt.plot(real_price, label=f'Real Price ({ticker})', color='blue')
    plt.plot(adjusted_predicted_price, label=f'Adjusted Predicted Price', color='green', linestyle='--')
    plt.title(f"{ticker} Stock Prediction (Hybrid RNN) - Corrected")
    plt.xlabel('Time (Test Data Index)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    plot_filename = f"validation_plot_{ticker}_{timestamp}.png"
    plt.savefig(plot_filename)
    print(f"Validation plot saved to {plot_filename}")
    plt.close() # Close the figure to free memory

# ---------------------------------------------------------
# 5. 메인 실행 블록 (대규모 수정)
# ---------------------------------------------------------
def main():
    model_name = 'Hybrid RNN'
    log_file = 'results.log'
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')

    # 1. 데이터 처리기 생성 및 종목 분할
    processor = StockDataProcessor(CONFIG['start_date'], CONFIG['end_date'])
    train_tickers, test_tickers = processor.get_and_split_tickers(test_size=CONFIG['test_split_ratio'])

    # 2. 학습 데이터셋 구축
    print("\nBuilding training dataset...")
    X_train_seq_np, y_train_np, X_train_static_np = processor.build_dataset(train_tickers, CONFIG['seq_length'])

    # 3. 테스트 데이터셋 구축
    print("\nBuilding testing dataset...")
    X_test_seq_np, y_test_np, X_test_static_np = processor.build_dataset(test_tickers, CONFIG['seq_length'])
    
    # 4. 텐서 변환
    X_train_seq = torch.tensor(X_train_seq_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_train_static = torch.tensor(X_train_static_np, dtype=torch.float32)

    X_test_seq = torch.tensor(X_test_seq_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    X_test_static = torch.tensor(X_test_static_np, dtype=torch.float32)

    print(f"\nFinal Train shapes: Seq={X_train_seq.shape}, Static={X_train_static.shape}")
    print(f"Final Test shapes: Seq={X_test_seq.shape}, Static={X_test_static.shape}")

    # 5. 모델 생성
    model = HybridRNN(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim'],
        num_layers=CONFIG['num_layers'],
        static_input_dim=CONFIG['static_input_dim'],
        static_hidden_dim=CONFIG['static_hidden_dim']
    )
    print(model)

    # 6. 학습 및 시간 측정
    start_time = time.time()
    loss_history = train_model(model, X_train_seq, X_train_static, y_train, CONFIG)
    end_time = time.time()
    training_time = end_time - start_time
    final_loss = loss_history[-1] if loss_history else float('inf')
    
    # 7. 전체 테스트셋 평가
    _, _, test_rmse = evaluate_model(model, X_test_seq, X_test_static, y_test, processor)
    
    print(f"\n--- Overall Evaluation ---")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Final Training Loss: {final_loss:.6f}")
    print(f"Overall Test Set RMSE: {test_rmse:.4f}")
    
    # 8. 결과 로깅
    log_results(log_file, timestamp, model_name, training_time, final_loss, test_rmse)

    # 9. [추가] 2개 종목 무작위 선정 및 개별 검증
    if len(test_tickers) >= 2:
        validation_tickers = random.sample(test_tickers, 2)
        for ticker in validation_tickers:
            validate_and_plot_ticker(model, ticker, processor, CONFIG, timestamp)
    else:
        print("\nNot enough test tickers to perform individual validation.")

if __name__ == "__main__":
    main()
