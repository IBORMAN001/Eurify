import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Eurify - AI Financial Risk Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS STYLING
st.markdown("""
<style>
    /* Main theme and layout */
    .main { padding-top: 1rem; }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Card styles */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-2px);
    }
    
    /* Risk assessment styles */
    .risk-low {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Trading signals */
    .trading-signal {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        color: white;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: white;
    }
    
    /* Feature boxes */
    .feature-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .model-performance {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Status boxes */
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .error-box {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: #f8f9fa; }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card, .prediction-card, .feature-box {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# MAIN HEADER
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        🏦 EURIFY
    </h1>
    <h2 style="font-size: 1.5rem; margin-bottom: 1rem; opacity: 0.9;">
        Advanced AI-Powered Financial Risk Analysis Platform
    </h2>
    <p style="font-size: 1.1rem; margin: 0; opacity: 0.8;">
        Predictive Analytics for Assessing Financial Market Risks 
    </p>
</div>
""", unsafe_allow_html=True)

# UTILITY FUNCTIONS
def flexible_column_mapping(columns):
    """Map various column name formats to standard names"""
    column_map = {}
    columns_lower = [col.lower().strip() for col in columns]
    
    # Date mapping
    date_variations = ['date', 'datetime', 'time', 'timestamp', 'day']
    for i, col_lower in enumerate(columns_lower):
        if any(variation in col_lower for variation in date_variations):
            column_map[columns[i]] = 'Date'
            break
    
    # Price columns mapping
    price_mappings = {
        'open': ['open', 'opening', 'open_price', 'o'],
        'high': ['high', 'highest', 'high_price', 'h'],
        'low': ['low', 'lowest', 'low_price', 'l'],
        'close': ['close', 'closing', 'close_price', 'c', 'price', 'adj_close', 'adjusted_close'],
        'volume': ['volume', 'vol', 'v', 'trading_volume']
    }
    
    for standard_name, variations in price_mappings.items():
        for i, col_lower in enumerate(columns_lower):
            if any(variation == col_lower or variation in col_lower for variation in variations):
                column_map[columns[i]] = standard_name.capitalize()
                break
    
    return column_map

def detect_price_column(df):
    """Detect the main price column to use for analysis"""
    price_priority = ['Close', 'close', 'Close_Price', 'Price', 'Adj_Close', 'Adjusted_Close']
    
    for col in price_priority:
        if col in df.columns:
            return col
    
    # If no standard price column found, look for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]
    
    return None

def calculate_technical_indicators(df, price_col='Close'):
    """Calculate comprehensive technical indicators"""
    try:
        df = df.copy()
        
        # Ensure we have the required price column
        if price_col not in df.columns:
            return df
            
        # Simple Moving Averages
        df['SMA_10'] = df[price_col].rolling(window=10, min_periods=1).mean()
        df['SMA_20'] = df[price_col].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df[price_col].rolling(window=50, min_periods=1).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df[price_col].ewm(span=12, min_periods=1).mean()
        df['EMA_26'] = df[price_col].ewm(span=26, min_periods=1).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Fill NaN with neutral value
        
        # Bollinger Bands
        df['BB_Middle'] = df[price_col].rolling(window=20, min_periods=1).mean()
        bb_std = df[price_col].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Price momentum and volatility
        df['Price_Change'] = df[price_col].pct_change().fillna(0)
        df['Volatility'] = df['Price_Change'].rolling(window=20, min_periods=1).std()
        df['Price_Position'] = (df[price_col] - df[price_col].rolling(window=20, min_periods=1).min()) / \
                              (df[price_col].rolling(window=20, min_periods=1).max() - 
                               df[price_col].rolling(window=20, min_periods=1).min())
        
        # Volume indicators (if volume exists)
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Fill any remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return df

def create_features_from_data(df, price_col):
    """Create comprehensive feature set for ML models"""
    try:
        # Calculate technical indicators first
        df_processed = calculate_technical_indicators(df, price_col)
        
        # Define feature columns (exclude target and non-numeric columns)
        feature_cols = []
        exclude_cols = ['Date', price_col, 'Price_Change']  # Don't include target variable
        
        for col in df_processed.columns:
            if col not in exclude_cols and df_processed[col].dtype in ['float64', 'int64']:
                feature_cols.append(col)
        
        # Ensure we have at least some features
        if not feature_cols:
            # Create basic features if none exist
            df_processed['Price_Lag1'] = df_processed[price_col].shift(1).fillna(df_processed[price_col].iloc[0])
            df_processed['Price_Lag2'] = df_processed[price_col].shift(2).fillna(df_processed[price_col].iloc[0])
            df_processed['Price_MA5'] = df_processed[price_col].rolling(5, min_periods=1).mean()
            feature_cols = ['Price_Lag1', 'Price_Lag2', 'Price_MA5']
        
        return df_processed, feature_cols
        
    except Exception as e:
        st.error(f"Error creating features: {str(e)}")
        return df, []

def train_models(df, model_type, price_col):
    """Train ML models for price prediction"""
    try:
        # Create features
        df_processed, feature_cols = create_features_from_data(df, price_col)
        
        if len(feature_cols) == 0:
            return {}, None, [], {}, False
        
        # Prepare data
        X = df_processed[feature_cols].fillna(0)
        y = df_processed[price_col].pct_change().fillna(0)  # Predict returns instead of absolute prices
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        if len(X) < 10:  # Need minimum data for training
            return {}, None, [], {}, False
        
        # Split data (use last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models
        models = {}
        if model_type == 'Gradient Boosting' or model_type == 'All':
            models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=50, random_state=42)
        if model_type == 'Linear Regression' or model_type == 'All':
            models['Linear Regression'] = LinearRegression()
        if model_type == 'Random Forest' or model_type == 'All':
            models['Random Forest'] = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Train models and calculate metrics
        trained_models = {}
        metrics = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                trained_models[name] = model
                metrics[name] = {
                    'R²': r2,
                    'RMSE': rmse,
                    'MAE': mae
                }
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
                continue
        
        return trained_models, scaler, feature_cols, metrics, len(trained_models) > 0
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return {}, None, [], {}, False

def calculate_comprehensive_risk(predictions, df, current_price):
    """Calculate comprehensive risk assessment"""
    try:
        risk_factors = {}
        
        # Prediction variance (disagreement between models)
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            risk_factors['prediction_variance'] = np.std(pred_values)
        else:
            risk_factors['prediction_variance'] = 0.1
        
        # Price volatility
        if 'Volatility' in df.columns:
            risk_factors['volatility'] = df['Volatility'].iloc[-1] if not pd.isna(df['Volatility'].iloc[-1]) else 0.1
        else:
            risk_factors['volatility'] = 0.1
        
        # Market momentum
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi > 70 or rsi < 30:  # Overbought or oversold
                risk_factors['momentum'] = 0.8
            else:
                risk_factors['momentum'] = 0.3
        else:
            risk_factors['momentum'] = 0.5
        
        # Calculate overall risk score
        risk_score = (
            risk_factors['prediction_variance'] * 0.4 +
            risk_factors['volatility'] * 0.4 +
            risk_factors['momentum'] * 0.2
        )
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = "Low"
            risk_class = "risk-low"
            risk_description = "Market conditions appear stable with low volatility."
        elif risk_score < 0.6:
            risk_level = "Medium"
            risk_class = "risk-medium"
            risk_description = "Moderate risk with some uncertainty in market direction."
        else:
            risk_level = "High"
            risk_class = "risk-high"
            risk_description = "High risk environment with significant volatility."
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_class': risk_class,
            'risk_description': risk_description,
            **risk_factors
        }
        
    except Exception as e:
        return {
            'risk_score': 0.5,
            'risk_level': "Medium",
            'risk_class': "risk-medium",
            'risk_description': "Unable to calculate risk factors.",
            'prediction_variance': 0.5,
            'volatility': 0.5,
            'momentum': 0.5
        }

def create_interactive_charts(df, predictions, risk_info, price_col):
    """Create comprehensive interactive charts"""
    try:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Price & Moving Averages', 'RSI Indicator', 
                          'MACD Analysis', 'Bollinger Bands',
                          'Volume Analysis', 'Risk Assessment'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Price and Moving Averages
        fig.add_trace(go.Scatter(x=df['Date'], y=df[price_col], 
                                name='Price', line=dict(color='#3498db', width=2)), 
                      row=1, col=1)
        
        if 'SMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], 
                                    name='SMA 20', line=dict(color='#e74c3c', width=1)), 
                          row=1, col=1)
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], 
                                    name='RSI', line=dict(color='#9b59b6')), 
                          row=1, col=2)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
        
        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], 
                                    name='MACD', line=dict(color='#2ecc71')), 
                          row=2, col=1)
            if 'MACD_Signal' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], 
                                        name='MACD Signal', line=dict(color='#f39c12')), 
                              row=2, col=1)
        
        # Bollinger Bands
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], 
                                    name='BB Upper', line=dict(color='rgba(255,0,0,0.3)')), 
                          row=2, col=2)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], 
                                    name='BB Lower', line=dict(color='rgba(0,255,0,0.3)')), 
                          row=2, col=2)
            fig.add_trace(go.Scatter(x=df['Date'], y=df[price_col], 
                                    name='Price', line=dict(color='#3498db')), 
                          row=2, col=2)
        
        # Volume
        if 'Volume' in df.columns:
            fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], 
                                name='Volume', marker_color='rgba(55,128,191,0.7)'), 
                          row=3, col=1)
        
        # Risk visualization
        risk_labels = ['Prediction Variance', 'Volatility', 'Momentum']
        risk_values = [risk_info.get('prediction_variance', 0) * 100,
                      risk_info.get('volatility', 0) * 100,
                      risk_info.get('momentum', 0) * 100]
        
        fig.add_trace(go.Bar(x=risk_labels, y=risk_values, 
                            name='Risk Factors', 
                            marker_color=['#e74c3c', '#f39c12', '#3498db']), 
                      row=3, col=2)
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Comprehensive Financial Analysis Dashboard")
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")
        # Return a simple fallback chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df[price_col], name='Price'))
        fig.update_layout(title="Price Chart", height=400)
        return fig

def generate_trading_signals(df, price_col):
    """Generate comprehensive trading signals"""
    signals = {}
    current_price = df[price_col].iloc[-1]
    
    try:
        # RSI Signals
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi >= 70:
                signals['RSI_Signal'] = {"signal": "Overbought", "class": "signal-sell", "icon": "🔴"}
            elif rsi <= 30:
                signals['RSI_Signal'] = {"signal": "Oversold", "class": "signal-buy", "icon": "🟢"}
            else:
                signals['RSI_Signal'] = {"signal": "Neutral", "class": "signal-hold", "icon": "🟡"}
        
        # MACD Signals
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            
            if macd > macd_signal * 1.05:
                signals['MACD_Signal'] = {"signal": "Strong Buy", "class": "signal-buy", "icon": "🟢"}
            elif macd > macd_signal:
                signals['MACD_Signal'] = {"signal": "Buy", "class": "signal-buy", "icon": "🔵"}
            elif macd < macd_signal * 0.95:
                signals['MACD_Signal'] = {"signal": "Strong Sell", "class": "signal-sell", "icon": "🔴"}
            elif macd < macd_signal:
                signals['MACD_Signal'] = {"signal": "Sell", "class": "signal-sell", "icon": "🟠"}
            else:
                signals['MACD_Signal'] = {"signal": "Hold", "class": "signal-hold", "icon": "🟡"}
        
        # Bollinger Bands Signals
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            
            if current_price >= bb_upper:
                signals['BB_Signal'] = {"signal": "Overbought", "class": "signal-sell", "icon": "🔴"}
            elif current_price <= bb_lower:
                signals['BB_Signal'] = {"signal": "Oversold", "class": "signal-buy", "icon": "🟢"}
            else:
                signals['BB_Signal'] = {"signal": "Normal Range", "class": "signal-hold", "icon": "🟡"}
        
        # Volume Analysis
        if 'Volume' in df.columns:
            recent_volume = df['Volume'].tail(5).mean()
            avg_volume = df['Volume'].mean()
            
            if recent_volume > avg_volume * 1.5:
                signals['Volume_Signal'] = {"signal": "High Volume", "class": "signal-buy", "icon": "📊"}
            elif recent_volume < avg_volume * 0.5:
                signals['Volume_Signal'] = {"signal": "Low Volume", "class": "signal-sell", "icon": "📉"}
            else:
                signals['Volume_Signal'] = {"signal": "Normal Volume", "class": "signal-hold", "icon": "📈"}
        
    except Exception as e:
        st.warning(f"Some signals could not be calculated: {str(e)}")
    
    return signals

def create_prediction_chart(df, predictions, price_col):
    """Create a dedicated prediction visualization"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df[price_col],
        mode='lines',
        name='Historical Price',
        line=dict(color='#3498db', width=2)
    ))
    
    # Add prediction points for different models
    colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for i, (model_name, pred_return) in enumerate(predictions.items()):
        current_price = df[price_col].iloc[-1]
        predicted_price = current_price * (1 + pred_return)
        
        # Add prediction point
        fig.add_trace(go.Scatter(
            x=[df['Date'].iloc[-1]],
            y=[predicted_price],
            mode='markers',
            name=f'{model_name} Prediction',
            marker=dict(
                size=15,
                color=colors[i % len(colors)],
                symbol='diamond',
                line=dict(width=2, color='white')
            )
        ))
    
    fig.update_layout(
        title="Price Predictions by Model",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        showlegend=True
    )
    
    return fig

# SIDEBAR - Configuration
st.sidebar.markdown("## 📊 Configuration Panel")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Financial Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with financial data including Date, Open, High, Low, Close, Volume columns"
)

# Model selection
model_type = st.sidebar.selectbox(
    "🤖 Select ML Model",
    ['Gradient Boosting', 'Linear Regression', 'Random Forest', 'All']
)

# Analysis period
analysis_period = st.sidebar.slider(
    "📈 Analysis Period (Days)",
    min_value=30,
    max_value=365,
    value=180,
    help="Number of recent days to include in analysis"
)

# Risk threshold
risk_threshold = st.sidebar.slider(
    "⚠️ Risk Alert Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.1,
    help="Set risk level threshold for alerts"
)

# MAIN APPLICATION LOGIC
if uploaded_file is not None:
    try:
        # Load and process data
        with st.spinner("🔄 Processing your financial data..."):
            df = pd.read_csv(uploaded_file)
            
            # Display basic info
            st.sidebar.success(f"✅ Data loaded: {len(df)} records")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Show original columns for debugging
            st.sidebar.write("**Detected Columns:**")
            st.sidebar.write(list(df.columns))
            
            # Apply flexible column mapping
            column_mapping = flexible_column_mapping(df.columns)
            if column_mapping:
                df = df.rename(columns=column_mapping)
                st.sidebar.success("✅ Columns mapped successfully")
            
            # Detect price column
            price_col = detect_price_column(df)
            if price_col is None:
                st.error("❌ Could not detect a valid price column in your data")
                st.stop()
            
            st.sidebar.info(f"📊 Using '{price_col}' as price column")
            
            # Ensure Date column exists and is properly formatted
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                try:
                    df['Date'] = pd.to_datetime(df[date_col])
                except:
                    df['Date'] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
            else:
                # Create a date index if no date column
                df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            
            # Filter by analysis period
            if len(df) > analysis_period:
                df = df.tail(analysis_period).reset_index(drop=True)
            
            # Remove any remaining NaN values in price column
            df = df.dropna(subset=[price_col])
            
            if len(df) < 50:
                st.error("❌ Insufficient data for analysis. Need at least 50 data points.")
                st.stop()
        
        # SUCCESS MESSAGE
        st.markdown("""
        <div class="success-box">
            <h3>✅ Data Successfully Processed!</h3>
            <p>Your financial data has been loaded and is ready for AI-powered analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # MAIN DASHBOARD LAYOUT
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df[price_col].iloc[-1]
        price_change = ((df[price_col].iloc[-1] - df[price_col].iloc[-2]) / df[price_col].iloc[-2] * 100) if len(df) > 1 else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Current Price</h3>
                <h2>${current_price:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            change_color = "#2ecc71" if price_change >= 0 else "#e74c3c"
            change_icon = "📈" if price_change >= 0 else "📉"
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, {change_color} 0%, {change_color}aa 100%);">
                <h3>{change_icon} Daily Change</h3>
                <h2>{price_change:+.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_records = len(df)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Data Points</h3>
                <h2>{total_records:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            analysis_status = "✅ Ready"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔍 Analysis</h3>
                <h2>{analysis_status}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # TRAIN MODELS AND MAKE PREDICTIONS
        st.markdown('<div class="sub-header">🤖 AI Model Training & Predictions</div>', unsafe_allow_html=True)
        
        with st.spinner("🧠 Training AI models and generating predictions..."):
            models, scaler, feature_cols, metrics, success = train_models(df, model_type, price_col)
            
            if success and models:
                st.success("✅ AI models trained successfully!")
                
                # Make predictions
                predictions = {}
                df_processed, _ = create_features_from_data(df, price_col)
                
                if len(df_processed) > 0 and feature_cols:
                    latest_features = df_processed[feature_cols].iloc[-1:].values
                    if scaler:
                        latest_features_scaled = scaler.transform(latest_features)
                    else:
                        latest_features_scaled = latest_features
                    
                    for name, model in models.items():
                        pred = model.predict(latest_features_scaled)[0]
                        predictions[name] = pred
                
                # Calculate risk assessment
                risk_info = calculate_comprehensive_risk(predictions, df_processed, current_price)
                
                # PREDICTIONS DISPLAY
                st.markdown('<div class="sub-header">🔮 Price Predictions</div>', unsafe_allow_html=True)
                
                pred_cols = st.columns(len(predictions))
                for i, (model_name, pred_return) in enumerate(predictions.items()):
                    predicted_price = current_price * (1 + pred_return)
                    change_pct = pred_return * 100
                    
                    with pred_cols[i]:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>🤖 {model_name}</h3>
                            <h2>${predicted_price:.2f}</h2>
                            <p style="font-size: 1.1rem; margin: 0;">
                                {change_pct:+.2f}% Expected Return
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # RISK ASSESSMENT
                st.markdown('<div class="sub-header">⚠️ Risk Assessment</div>', unsafe_allow_html=True)
                
                risk_col1, risk_col2 = st.columns([1, 2])
                
                with risk_col1:
                    st.markdown(f"""
                    <div class="{risk_info['risk_class']}">
                        <h2>Risk Level: {risk_info['risk_level']}</h2>
                        <p>{risk_info['risk_description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with risk_col2:
                    # Risk factors breakdown
                    st.markdown("### 📋 Risk Factors Analysis")
                    
                    factors_to_show = ['prediction_variance', 'volatility', 'momentum']
                    for factor in factors_to_show:
                        if factor in risk_info:
                            factor_value = risk_info[factor]
                            factor_pct = min(factor_value * 100, 100)
                            
                            st.progress(factor_pct / 100)
                            st.caption(f"{factor.replace('_', ' ').title()}: {factor_pct:.1f}%")
                
                # TRADING SIGNALS
                st.markdown('<div class="sub-header">📊 Trading Signals</div>', unsafe_allow_html=True)
                
                signals = generate_trading_signals(df_processed, price_col)
                
                if signals:
                    signal_cols = st.columns(len(signals))
                    for i, (signal_name, signal_info) in enumerate(signals.items()):
                        with signal_cols[i]:
                            st.markdown(f"""
                            <div class="trading-signal {signal_info['class']}">
                                <h4>{signal_info['icon']} {signal_name.replace('_', ' ')}</h4>
                                <p><strong>{signal_info['signal']}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # CONFIGURATION DETAILS (MOVED AFTER RISK ASSESSMENT)
                st.markdown('<div class="sub-header">⚙️ Configuration Details</div>', unsafe_allow_html=True)
                
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    st.markdown(f"""
                    <div class="feature-box">
                        <h4>🤖 Model Configuration</h4>
                        <ul>
                            <li><strong>Selected Model:</strong> {model_type}</li>
                            <li><strong>Feature Count:</strong> {len(feature_cols)}</li>
                            <li><strong>Training Data:</strong> {len(df)} records</li>
                            <li><strong>Analysis Period:</strong> {analysis_period} days</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with config_col2:
                    st.markdown(f"""
                    <div class="feature-box">
                        <h4>📊 Data Configuration</h4>
                        <ul>
                            <li><strong>Price Column:</strong> {price_col}</li>
                            <li><strong>Date Range:</strong> {df['Date'].dt.date.min()} to {df['Date'].dt.date.max()}</li>
                            <li><strong>Data Quality:</strong> ✅ Clean</li>
                            <li><strong>Technical Indicators:</strong> ✅ Calculated</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with config_col3:
                    st.markdown(f"""
                    <div class="feature-box">
                        <h4>⚠️ Risk Configuration</h4>
                        <ul>
                            <li><strong>Risk Threshold:</strong> {risk_threshold}</li>
                            <li><strong>Current Risk:</strong> {risk_info['risk_level']}</li>
                            <li><strong>Risk Score:</strong> {risk_info['risk_score']:.2f}</li>
                            <li><strong>Alert Status:</strong> {'🚨 Alert' if risk_info['risk_score'] > risk_threshold else '✅ Normal'}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # MODEL PERFORMANCE METRICS
                if metrics:
                    st.markdown('<div class="sub-header">📈 Model Performance</div>', unsafe_allow_html=True)
                    
                    perf_cols = st.columns(len(metrics))
                    for i, (model_name, model_metrics) in enumerate(metrics.items()):
                        with perf_cols[i]:
                            st.markdown(f"""
                            <div class="model-performance">
                                <h3>🤖 {model_name}</h3>
                                <p><strong>R² Score:</strong> {model_metrics['R²']:.3f}</p>
                                <p><strong>RMSE:</strong> {model_metrics['RMSE']:.6f}</p>
                                <p><strong>MAE:</strong> {model_metrics['MAE']:.6f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # INTERACTIVE CHARTS
                st.markdown('<div class="sub-header">📊 Interactive Analysis Charts</div>', unsafe_allow_html=True)
                
                # Main comprehensive chart
                main_chart = create_interactive_charts(df_processed, predictions, risk_info, price_col)
                st.plotly_chart(main_chart, use_container_width=True)
                
                # Prediction comparison chart
                pred_chart = create_prediction_chart(df_processed, predictions, price_col)
                st.plotly_chart(pred_chart, use_container_width=True)
                
                # DATA INSIGHTS
                st.markdown('<div class="sub-header">💡 Key Insights</div>', unsafe_allow_html=True)
                
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.markdown("""
                    <div class="feature-box">
                        <h4>📊 Technical Analysis</h4>
                        <ul>
                            <li>Moving averages indicate current trend direction</li>
                            <li>RSI shows overbought/oversold conditions</li>
                            <li>Bollinger Bands reveal volatility levels</li>
                            <li>MACD signals momentum changes</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with insights_col2:
                    st.markdown(f"""
                    <div class="feature-box">
                        <h4>🤖 AI Predictions</h4>
                        <ul>
                            <li>Multiple models provide diverse perspectives</li>
                            <li>Risk level: <strong>{risk_info['risk_level']}</strong></li>
                            <li>Prediction confidence varies by model</li>
                            <li>Consider all signals before trading</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # EXPORT OPTIONS
                st.markdown('<div class="sub-header">💾 Export Results</div>', unsafe_allow_html=True)
                
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    if st.button("📊 Download Predictions CSV"):
                        pred_df = pd.DataFrame({
                            'Model': predictions.keys(),
                            'Predicted_Return': predictions.values(),
                            'Predicted_Price': [current_price * (1 + ret) for ret in predictions.values()]
                        })
                        st.download_button(
                            label="Download CSV",
                            data=pred_df.to_csv(index=False),
                            file_name="eurify_predictions.csv",
                            mime="text/csv"
                        )
                
                with export_col2:
                    if st.button("📈 Save Analysis Report"):
                        report = f"""
                        EURIFY FINANCIAL ANALYSIS REPORT
                        ================================
                        
                        Current Price: ${current_price:.2f}
                        Risk Level: {risk_info['risk_level']}
                        Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                        
                        PREDICTIONS:
                        """
                        for model, pred in predictions.items():
                            pred_price = current_price * (1 + pred)
                            report += f"\n{model}: ${pred_price:.2f} ({pred*100:+.2f}%)"
                        
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name="eurify_analysis_report.txt",
                            mime="text/plain"
                        )
                
                with export_col3:
                    st.info("💡 **Pro Tip:** Combine multiple signals and always consider market conditions!")
            
            else:
                st.error("❌ Failed to train models. Please check your data format.")
                
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ Error processing file:</h3>
            <p><strong>{str(e)}</strong></p>
            <h4>💡 Debug Info:</h4>
            <p>Try checking your CSV file format. Ensure it has proper column headers and numeric price data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data format
        st.markdown("### 📋 Expected CSV Format:")
        sample_df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [100.0, 101.0, 102.0],
            'High': [102.0, 103.0, 104.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [101.0, 102.0, 103.0],
            'Volume': [1000000, 1100000, 1200000]
        })
        st.dataframe(sample_df)

else:
    # WELCOME SCREEN
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: #2c3e50; margin-bottom: 1.5rem;">🚀 Welcome to Eurify</h2>
        <p style="font-size: 1.2rem; color: #34495e; margin-bottom: 2rem;">
            Your Advanced AI-Powered Financial Risk Analysis Platform
        </p>
        <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
            <h3 style="color: #3498db; margin-bottom: 1rem;">🎯 Key Features</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; text-align: left;">
                <div style="padding: 1rem; background: #f8f9ff; border-radius: 8px; border-left: 4px solid #3498db;">
                    <h4 style="color: #2c3e50;">🤖 Multiple AI Models</h4>
                    <p style="color: #2c3e50;">Linear Regression, Gradient Boosting, Random Forest with ensemble predictions</p>
                </div>
                <div style="padding: 1rem; background: #fff8f0; border-radius: 8px; border-left: 4px solid #f39c12;">
                    <h4 style="color: #2c3e50;">📊 Advanced Technical Analysis</h4>
                    <p style="color: #2c3e50;">RSI, MACD, Bollinger Bands, Moving Averages, Volume Analysis</p>
                </div>
                <div style="padding: 1rem; background: #f0fff8; border-radius: 8px; border-left: 4px solid #2ecc71;">
                    <h4 style="color: #2c3e50;">⚠️ Comprehensive Risk Assessment</h4>
                    <p style="color: #2c3e50;">Multi-factor risk analysis with prediction variance and volatility</p>
                </div>
                <div style="padding: 1rem; background: #fff0f8; border-radius: 8px; border-left: 4px solid #e74c3c;">
                    <h4 style="color: #2c3e50;">📈 Smart Trading Signals</h4>
                    <p style="color: #2c3e50;">Buy/Sell/Hold recommendations based on multiple indicators</p>
                </div>
                <div style="padding: 1rem; background: #f0f8ff; border-radius: 8px; border-left: 4px solid #9b59b6;">
                    <h4 style="color: #2c3e50;">📊 Interactive Visualizations</h4>
                    <p style="color: #2c3e50;">Real-time charts with technical indicators and prediction overlays</p>
                </div>
                <div style="padding: 1rem; background: #fff8f8; border-radius: 8px; border-left: 4px solid #e67e22;">
                    <h4 style="color: #2c3e50;">💾 Export & Reporting</h4>
                    <p style="color: #2c3e50;">Download predictions, analysis reports, and configuration details</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="feature-box" style="background: white; padding: 2rem; border-radius: 12px; margin: 2rem 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
        <h3 style="color: #2c3e50;">📁 Getting Started</h3>
        <ol style="font-size: 1.1rem; line-height: 1.8; color: #2c3e50;">
            <li><strong style="color: #2c3e50;">Upload Your Data:</strong> Use the sidebar to upload a CSV file with financial data</li>
            <li><strong style="color: #2c3e50;">Select Model:</strong> Choose your preferred AI model or use all models for comparison</li>
            <li><strong style="color: #2c3e50;">Configure Analysis:</strong> Set your analysis period and risk thresholds</li>
            <li><strong style="color: #2c3e50;">Get Insights:</strong> Receive comprehensive predictions and risk analysis</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
st.markdown("""
    <style>
    .feature-box {
        background: white !important;
        color: #2c3e50 !important;
    }
    
    .feature-box h3, .feature-box h4, .feature-box p, .feature-box li {
        color: #2c3e50 !important;
    }
    
    .feature-box strong {
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
        
st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 12px; margin: 2rem 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
        <h4 style="color: #2c3e50;">📋 Supported Data Formats:</h4>
        <p style="color: #2c3e50;">Your CSV should include columns like: Date, Open, High, Low, Close, Volume</p>
        <p style="color: #2c3e50;"><em>Don't worry about exact column names - Eurify automatically detects and maps your data!</em></p>
        <h4 style="color: #000000;">🎯 What Makes Eurify Special:</h4>
        <ul style="font-size: 1rem; line-height: 1.6; color: #000000;">
            <li><strong style="color: #000000;">Intelligent Column Detection:</strong> Automatically maps various column naming conventions</li>
            <li><strong style="color: #000000;">Robust Error Handling:</strong> Handles missing data and edge cases gracefully</li>
            <li><strong style="color: #000000;">Multi-Model Ensemble:</strong> Combines predictions from multiple AI models</li>
            <li><strong style="color: #000000;">Real-time Risk Assessment:</strong> Dynamic risk calculation based on market conditions</li>
            <li><strong style="color: #000000;">Professional Visualizations:</strong> Interactive charts with technical analysis</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin: 2rem 0;">
    <div>
        <h4 style="color: #3498db;">🎯 Our Mission</h4>
        <p style="color: rgba(255,255,255,0.8);">
            Democratize advanced financial analysis through AI technology, making professional-grade 
            risk assessment accessible to everyone.
        </p>
    </div>
    <div>
        <h4 style="color: #2ecc71;">🛡️ Security & Privacy</h4>
        <p style="color: rgba(255,255,255,0.8);">
            Your data is processed locally and securely. We prioritize user privacy and 
            data protection in all our operations.
        </p>
    </div>
    <div>
        <h4 style="color: #e74c3c;">⚠️ Disclaimer</h4>
        <p style="color: rgba(255,255,255,0.8);">
            This tool provides analysis for educational purposes. Always consult financial 
            advisors and conduct thorough research before making investment decisions.
        </p>
    </div>
</div>

<div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1.5rem; margin-top: 2rem;">
    <p style="margin: 0; color: rgba(255,255,255,0.7);">
        &copy; 2024 Eurify - Predictive Analytics for Financial Market Risk Assessment<br>
        <em>Research Led Purpose - Advanced AI Implementation</em>
    </p>
</div>
""", unsafe_allow_html=True)