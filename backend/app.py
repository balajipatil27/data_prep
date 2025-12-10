import os
import pandas as pd
import numpy as np
import json
import sqlite3
import joblib
import traceback
import warnings
import io
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, session, make_response
from flask_cors import CORS
from flask_session import Session
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, silhouette_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Setup CORS - allow all for simplicity, update for production
CORS(app, resources={r"/api/*": {"origins": "*"}})
Session(app)

# Create directories
def create_directories():
    directories = ['uploads', 'processed', 'tmp']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('datasets.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            original_filename TEXT,
            processed_filename TEXT,
            upload_time TIMESTAMP,
            user_session TEXT,
            preprocessing_steps TEXT,
            results TEXT
        )
    ''')
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_dataset(df):
    """Analyze dataset and return statistics"""
    analysis = {
        'shape': df.shape,
        'columns': [],
        'missing_percentage': {},
        'data_types': {},
        'numerical_cols': [],
        'categorical_cols': [],
        'sample_data': {}
    }
    
    for col in df.columns:
        missing_percent = (df[col].isnull().sum() / len(df)) * 100
        analysis['missing_percentage'][col] = round(missing_percent, 2)
        analysis['data_types'][col] = str(df[col].dtype)
        
        if pd.api.types.is_numeric_dtype(df[col]):
            analysis['numerical_cols'].append(col)
        else:
            analysis['categorical_cols'].append(col)
            
        analysis['columns'].append({
            'name': col,
            'type': str(df[col].dtype),
            'missing_percent': round(missing_percent, 2),
            'unique_values': int(df[col].nunique()) if not pd.api.types.is_numeric_dtype(df[col]) else None,
            'sample_values': df[col].dropna().head(5).tolist() if df[col].dropna().shape[0] > 0 else []
        })
    
    # Add some sample rows
    analysis['sample_data']['head'] = df.head(5).fillna('NaN').to_dict('records')
    
    return analysis

def preprocess_dataset(df, preprocessing_options):
    """Apply preprocessing based on user options"""
    steps = []
    original_shape = df.shape
    
    # Step 1: Remove columns with >50% missing values
    cols_to_drop = []
    for col in df.columns:
        missing_percent = (df[col].isnull().sum() / len(df)) * 100
        if missing_percent > 50:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        steps.append(f"Removed columns with >50% missing values: {', '.join(cols_to_drop)}")
    
    # Step 2: Remove duplicate rows
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates()
        steps.append(f"Removed {duplicates_count} duplicate rows")
    
    # Step 3: Handle data type conversions
    if 'type_conversions' in preprocessing_options:
        for col, new_type in preprocessing_options['type_conversions'].items():
            if col in df.columns:
                try:
                    if new_type == 'numeric':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        steps.append(f"Converted column '{col}' to numeric")
                    elif new_type == 'datetime':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        steps.append(f"Converted column '{col}' to datetime")
                    elif new_type == 'category':
                        df[col] = df[col].astype('category')
                        steps.append(f"Converted column '{col}' to category")
                except Exception as e:
                    steps.append(f"Failed to convert column '{col}' to {new_type}: {str(e)}")
    
    # Step 4: Handle missing values
    if 'missing_value_strategy' in preprocessing_options:
        for col, strategy in preprocessing_options['missing_value_strategy'].items():
            if col in df.columns and df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                    steps.append(f"Filled missing values in '{col}' with mean: {df[col].mean():.2f}")
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                    steps.append(f"Filled missing values in '{col}' with median: {df[col].median():.2f}")
                elif strategy == 'mode':
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                    df[col] = df[col].fillna(mode_val)
                    steps.append(f"Filled missing values in '{col}' with mode: {mode_val}")
                elif strategy == 'drop':
                    initial_len = len(df)
                    df = df.dropna(subset=[col])
                    dropped = initial_len - len(df)
                    steps.append(f"Dropped {dropped} rows with missing values in '{col}'")
    
    # Step 5: Handle categorical encoding
    if 'encoding' in preprocessing_options:
        for col, encoding_type in preprocessing_options['encoding'].items():
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                if encoding_type == 'label':
                    try:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str).fillna('missing'))
                        steps.append(f"Applied label encoding to '{col}'")
                    except Exception as e:
                        steps.append(f"Failed label encoding for '{col}': {str(e)}")
                elif encoding_type == 'onehot':
                    try:
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                        steps.append(f"Applied one-hot encoding to '{col}' (created {len(dummies.columns)} new columns)")
                    except Exception as e:
                        steps.append(f"Failed one-hot encoding for '{col}': {str(e)}")
    
    # Step 6: Handle outliers (IQR method for numerical columns)
    if preprocessing_options.get('remove_outliers', False):
        outlier_cols = []
        outliers_removed = 0
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                initial_len = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                removed = initial_len - len(df)
                if removed > 0:
                    outlier_cols.append(col)
                    outliers_removed += removed
        
        if outlier_cols:
            steps.append(f"Removed {outliers_removed} outliers from columns: {', '.join(outlier_cols)}")
    
    final_shape = df.shape
    steps.insert(0, f"Original dataset shape: {original_shape}")
    steps.append(f"Processed dataset shape: {final_shape}")
    
    # Ensure no NaN values remain for model training
    df = df.fillna(0)
    
    return df, steps

def detect_problem_type(df, target_column):
    """Detect if problem is classification or regression"""
    if target_column is None or target_column not in df.columns:
        return 'clustering'
    
    # Try to convert to numeric for analysis
    try:
        if pd.api.types.is_numeric_dtype(df[target_column]):
            unique_values = df[target_column].nunique()
            if unique_values <= 10:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'
    except:
        return 'classification'

def prepare_data_for_training(df, target_column, problem_type):
    """Prepare data for model training"""
    try:
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Convert non-numeric columns in X to numeric
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    except:
                        X[col] = pd.factorize(X[col])[0]
            
            # Convert y based on problem type
            if problem_type == 'classification':
                if not pd.api.types.is_numeric_dtype(y):
                    y = pd.factorize(y)[0]
            
            # Fill any remaining NaN values
            X = X.fillna(0)
            
            # Split data
            if len(X) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None
                )
                return X_train, X_test, y_train, y_test
            else:
                return None, None, None, None
        else:
            # For clustering, use all data
            X = df.copy()
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    except:
                        X[col] = pd.factorize(X[col])[0]
            X = X.fillna(0)
            return X, None, None, None
            
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return None, None, None, None

def train_models(X_train, X_test, y_train, y_test, problem_type):
    """Train and evaluate models"""
    results = {}
    
    if problem_type == 'classification' and y_train is not None:
        # Ensure we have at least 2 classes
        unique_classes = len(np.unique(y_train))
        if unique_classes < 2:
            return {'error': 'Need at least 2 classes for classification'}
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                if X_test is not None and len(X_test) > 0:
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    results[name] = {
                        'accuracy': round(float(accuracy), 4),
                        'type': 'classification'
                    }
                else:
                    results[name] = {
                        'accuracy': 'No test data',
                        'type': 'classification'
                    }
            except Exception as e:
                results[name] = {
                    'accuracy': 'Error',
                    'error': str(e),
                    'type': 'classification'
                }
    
    elif problem_type == 'regression' and y_train is not None:
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'SVM': SVR(kernel='linear')
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                if X_test is not None and len(X_test) > 0:
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    results[name] = {
                        'r2_score': round(float(r2), 4),
                        'mse': round(float(mse), 4),
                        'type': 'regression'
                    }
                else:
                    results[name] = {
                        'r2_score': 'No test data',
                        'mse': 'No test data',
                        'type': 'regression'
                    }
            except Exception as e:
                results[name] = {
                    'r2_score': 'Error',
                    'mse': 'Error',
                    'error': str(e),
                    'type': 'regression'
                }
    
    elif problem_type == 'clustering' and X_train is not None:
        try:
            # Determine optimal number of clusters (max 5 or data size)
            n_clusters = min(5, len(X_train))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X_train)
                silhouette = silhouette_score(X_train, clusters) if len(np.unique(clusters)) > 1 else 0
                results['K-Means'] = {
                    'silhouette_score': round(float(silhouette), 4),
                    'n_clusters': n_clusters,
                    'type': 'clustering'
                }
            else:
                results['K-Means'] = {
                    'silhouette_score': 'Insufficient data',
                    'type': 'clustering'
                }
        except Exception as e:
            results['K-Means'] = {
                'silhouette_score': 'Error',
                'error': str(e),
                'type': 'clustering'
            }
    
    return results

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Dataset Preprocessor API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """Handle dataset upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use CSV or Excel files.'}), 400
        
        # Generate unique ID
        dataset_id = str(uuid.uuid4())
        filename = secure_filename(f"{dataset_id}_{file.filename}")
        filepath = os.path.join('uploads', filename)
        
        # Save file
        file.save(filepath)
        
        # Read dataset
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        
        # Basic validation
        if df.empty:
            return jsonify({'error': 'Dataset is empty'}), 400
        
        # Analyze dataset
        analysis = analyze_dataset(df)
        
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'filename': filename,
            'analysis': analysis,
            'message': f'Dataset uploaded successfully. Shape: {df.shape}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_uploaded():
    """Analyze uploaded dataset"""
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join('uploads', filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Read dataset
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        analysis = analyze_dataset(df)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """Apply preprocessing to dataset"""
    try:
        data = request.json
        filename = data.get('filename')
        preprocessing_options = data.get('options', {})
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join('uploads', filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Read dataset
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Apply preprocessing
        processed_df, steps = preprocess_dataset(df, preprocessing_options)
        
        # Save processed dataset
        processed_filename = f"processed_{filename}"
        processed_filepath = os.path.join('processed', processed_filename)
        processed_df.to_csv(processed_filepath, index=False)
        
        # Analyze processed dataset
        processed_analysis = analyze_dataset(processed_df)
        
        return jsonify({
            'success': True,
            'processed_filename': processed_filename,
            'steps': steps,
            'processed_analysis': processed_analysis,
            'message': 'Preprocessing completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/train', methods=['POST'])
def train_and_compare():
    """Train models on original and processed data"""
    try:
        data = request.json
        original_filename = data.get('original_filename')
        processed_filename = data.get('processed_filename')
        target_column = data.get('target_column')
        
        if not original_filename:
            return jsonify({'error': 'No original filename provided'}), 400
        
        # Load original dataset
        original_filepath = os.path.join('uploads', original_filename)
        if not os.path.exists(original_filepath):
            return jsonify({'error': 'Original file not found'}), 404
        
        if original_filename.endswith('.csv'):
            original_df = pd.read_csv(original_filepath)
        else:
            original_df = pd.read_excel(original_filepath)
        
        results = {'original': {}, 'processed': {}}
        
        # Determine problem type
        problem_type = detect_problem_type(original_df, target_column)
        
        # Train on original data
        if target_column and target_column in original_df.columns:
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = prepare_data_for_training(
                original_df, target_column, problem_type
            )
            
            if X_train_orig is not None:
                results['original'] = train_models(
                    X_train_orig, X_test_orig, y_train_orig, y_test_orig, problem_type
                )
        
        # Train on processed data
        if processed_filename:
            processed_filepath = os.path.join('processed', processed_filename)
            if os.path.exists(processed_filepath):
                processed_df = pd.read_csv(processed_filepath)
                
                if target_column and target_column in processed_df.columns:
                    X_train_proc, X_test_proc, y_train_proc, y_test_proc = prepare_data_for_training(
                        processed_df, target_column, problem_type
                    )
                    
                    if X_train_proc is not None:
                        results['processed'] = train_models(
                            X_train_proc, X_test_proc, y_train_proc, y_test_proc, problem_type
                        )
        
        return jsonify({
            'success': True,
            'results': results,
            'problem_type': problem_type,
            'message': 'Model training completed'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed dataset"""
    try:
        filepath = os.path.join('processed', filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample', methods=['GET'])
def get_sample_datasets():
    """Get sample datasets for testing"""
    # Create a sample dataset
    np.random.seed(42)
    n_samples = 100
    
    sample_data = {
        'age': np.random.randint(18, 65, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'experience': np.random.randint(1, 30, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR'], n_samples),
        'salary': np.random.normal(75000, 20000, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    # Add some missing values
    for col in ['age', 'income', 'education']:
        idx = np.random.choice(n_samples, size=10, replace=False)
        for i in idx:
            sample_data[col][i] = np.nan
    
    df = pd.DataFrame(sample_data)
    sample_filepath = os.path.join('uploads', 'sample_dataset.csv')
    df.to_csv(sample_filepath, index=False)
    
    return jsonify({
        'success': True,
        'sample_filename': 'sample_dataset.csv',
        'message': 'Sample dataset created'
    })

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Dataset Preprocessor API',
        'endpoints': {
            'GET /api/health': 'Health check',
            'POST /api/upload': 'Upload dataset',
            'POST /api/preprocess': 'Preprocess dataset',
            'POST /api/train': 'Train models',
            'GET /api/download/<filename>': 'Download file',
            'GET /api/sample': 'Get sample dataset'
        }
    })

if __name__ == '__main__':
    # Create directories
    create_directories()
    init_db()
    
    # Get port from environment (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
