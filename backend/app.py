import os
import pandas as pd
import numpy as np
import json
import sqlite3
import joblib
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
from flask_session import Session
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, silhouette_score
import io
import uuid

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['DATABASE'] = 'dataset_preprocessor.db'

Session(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
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

def save_to_db(dataset_id, original_filename, processed_filename, preprocessing_steps, results):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO datasets (id, original_filename, processed_filename, upload_time, user_session, preprocessing_steps, results)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (dataset_id, original_filename, processed_filename, datetime.now(), session.sid, 
          json.dumps(preprocessing_steps), json.dumps(results)))
    conn.commit()
    conn.close()

def analyze_dataset(df):
    """Analyze dataset and return statistics"""
    analysis = {
        'shape': df.shape,
        'columns': [],
        'missing_percentage': {},
        'data_types': {},
        'numerical_cols': [],
        'categorical_cols': []
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
            'unique_values': df[col].nunique() if not pd.api.types.is_numeric_dtype(df[col]) else None
        })
    
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
                    elif new_type == 'datetime':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif new_type == 'category':
                        df[col] = df[col].astype('category')
                    steps.append(f"Converted column '{col}' to {new_type}")
                except Exception as e:
                    steps.append(f"Failed to convert column '{col}' to {new_type}: {str(e)}")
    
    # Step 4: Handle missing values
    if 'missing_value_strategy' in preprocessing_options:
        for col, strategy in preprocessing_options['missing_value_strategy'].items():
            if col in df.columns and df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
                elif strategy == 'drop':
                    df = df.dropna(subset=[col])
                steps.append(f"Filled missing values in '{col}' using {strategy}")
    
    # Step 5: Handle categorical encoding
    if 'encoding' in preprocessing_options:
        for col, encoding_type in preprocessing_options['encoding'].items():
            if col in df.columns and col in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
                if encoding_type == 'label':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    steps.append(f"Applied label encoding to '{col}'")
                elif encoding_type == 'onehot':
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    steps.append(f"Applied one-hot encoding to '{col}' (created {len(dummies.columns)} new columns)")
    
    # Step 6: Handle outliers (IQR method for numerical columns)
    if preprocessing_options.get('remove_outliers', False):
        outlier_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    outlier_cols.append(col)
        
        if outlier_cols:
            steps.append(f"Removed outliers from columns: {', '.join(outlier_cols)}")
    
    final_shape = df.shape
    steps.insert(0, f"Original dataset shape: {original_shape}")
    steps.append(f"Processed dataset shape: {final_shape}")
    
    return df, steps

def train_models(X_train, X_test, y_train, y_test, problem_type):
    """Train and evaluate models"""
    results = {}
    
    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Decision Tree': DecisionTreeClassifier(),
            'SVM': SVC(kernel='linear', probability=True)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'accuracy': round(accuracy, 4),
                    'type': 'classification'
                }
            except Exception as e:
                results[name] = {
                    'accuracy': 'Error',
                    'error': str(e),
                    'type': 'classification'
                }
    
    elif problem_type == 'regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100),
            'Decision Tree': DecisionTreeRegressor(),
            'SVM': SVR(kernel='linear')
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                results[name] = {
                    'r2_score': round(r2, 4),
                    'mse': round(mse, 4),
                    'type': 'regression'
                }
            except Exception as e:
                results[name] = {
                    'r2_score': 'Error',
                    'mse': 'Error',
                    'error': str(e),
                    'type': 'regression'
                }
    
    elif problem_type == 'clustering':
        try:
            # For clustering, we use silhouette score
            kmeans = KMeans(n_clusters=min(5, len(X_train)), random_state=42)
            clusters = kmeans.fit_predict(X_train)
            silhouette = silhouette_score(X_train, clusters)
            results['K-Means'] = {
                'silhouette_score': round(silhouette, 4),
                'n_clusters': min(5, len(X_train)),
                'type': 'clustering'
            }
        except Exception as e:
            results['K-Means'] = {
                'silhouette_score': 'Error',
                'error': str(e),
                'type': 'clustering'
            }
    
    return results

def detect_problem_type(df, target_column):
    """Detect if problem is classification or regression"""
    if target_column is None or target_column not in df.columns:
        return 'clustering'
    
    unique_values = df[target_column].nunique()
    if unique_values <= 10 or not pd.api.types.is_numeric_dtype(df[target_column]):
        return 'classification'
    else:
        return 'regression'

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
        
        # Generate unique ID for this dataset
        dataset_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(f"{dataset_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read dataset
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Analyze dataset
        analysis = analyze_dataset(df)
        
        # Store in session
        session['dataset_id'] = dataset_id
        session['original_filepath'] = filepath
        session['dataset_analysis'] = analysis
        
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'analysis': analysis,
            'message': f'Dataset uploaded successfully. Shape: {df.shape}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """Apply preprocessing to dataset"""
    try:
        data = request.json
        dataset_id = session.get('dataset_id')
        original_filepath = session.get('original_filepath')
        
        if not dataset_id or not original_filepath:
            return jsonify({'error': 'No dataset found. Please upload first.'}), 400
        
        # Read original dataset
        if original_filepath.endswith('.csv'):
            df = pd.read_csv(original_filepath)
        else:
            df = pd.read_excel(original_filepath)
        
        # Apply preprocessing
        processed_df, steps = preprocess_dataset(df, data)
        
        # Save processed dataset
        processed_filename = f"processed_{dataset_id}.csv"
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        processed_df.to_csv(processed_filepath, index=False)
        
        # Update session
        session['processed_filepath'] = processed_filepath
        session['preprocessing_steps'] = steps
        
        # Analyze processed dataset
        processed_analysis = analyze_dataset(processed_df)
        
        return jsonify({
            'success': True,
            'steps': steps,
            'processed_analysis': processed_analysis,
            'processed_filename': processed_filename,
            'message': f'Preprocessing completed. New shape: {processed_df.shape}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/train', methods=['POST'])
def train_and_compare():
    """Train models on original and processed data"""
    try:
        data = request.json
        target_column = data.get('target_column')
        models_to_use = data.get('models', ['all'])
        
        dataset_id = session.get('dataset_id')
        original_filepath = session.get('original_filepath')
        processed_filepath = session.get('processed_filepath')
        
        if not dataset_id:
            return jsonify({'error': 'No dataset found. Please upload first.'}), 400
        
        # Read datasets
        if original_filepath.endswith('.csv'):
            original_df = pd.read_csv(original_filepath)
        else:
            original_df = pd.read_excel(original_filepath)
        
        processed_df = pd.read_csv(processed_filepath) if processed_filepath else None
        
        results = {
            'original': {},
            'processed': {}
        }
        
        # Determine problem type
        problem_type = detect_problem_type(original_df, target_column)
        
        # Prepare data for original dataset
        if target_column and target_column in original_df.columns:
            X_original = original_df.drop(columns=[target_column])
            y_original = original_df[target_column]
            
            # Handle non-numeric data in X
            for col in X_original.columns:
                if not pd.api.types.is_numeric_dtype(X_original[col]):
                    X_original[col] = pd.factorize(X_original[col])[0]
            
            # Handle non-numeric data in y for classification
            if problem_type == 'classification' and not pd.api.types.is_numeric_dtype(y_original):
                y_original = pd.factorize(y_original)[0]
            
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                X_original, y_original, test_size=0.2, random_state=42
            )
            
            # Train models on original data
            results['original'] = train_models(
                X_train_orig, X_test_orig, y_train_orig, y_test_orig, problem_type
            )
        
        # Prepare data for processed dataset
        if processed_df is not None and target_column and target_column in processed_df.columns:
            X_processed = processed_df.drop(columns=[target_column])
            y_processed = processed_df[target_column]
            
            # Handle non-numeric data in y for classification
            if problem_type == 'classification' and not pd.api.types.is_numeric_dtype(y_processed):
                y_processed = pd.factorize(y_processed)[0]
            
            X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )
            
            # Train models on processed data
            results['processed'] = train_models(
                X_train_proc, X_test_proc, y_train_proc, y_test_proc, problem_type
            )
        
        # Save results to database
        if processed_filepath:
            save_to_db(
                dataset_id,
                os.path.basename(original_filepath),
                os.path.basename(processed_filepath),
                session.get('preprocessing_steps', []),
                results
            )
        
        return jsonify({
            'success': True,
            'results': results,
            'problem_type': problem_type,
            'message': 'Model training and comparison completed'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed dataset"""
    try:
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=f"processed_{filename}",
            mimetype='text/csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get list of processed datasets for current session"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, original_filename, processed_filename, upload_time 
            FROM datasets 
            WHERE user_session = ? 
            ORDER BY upload_time DESC
        ''', (session.sid,))
        datasets = cursor.fetchall()
        conn.close()
        
        return jsonify({
            'success': True,
            'datasets': [
                {
                    'id': row[0],
                    'original_filename': row[1],
                    'processed_filename': row[2],
                    'upload_time': row[3]
                }
                for row in datasets
            ]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    # Initialize database
    init_db()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
