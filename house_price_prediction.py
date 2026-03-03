"""
HOUSE PRICE PREDICTION USING MULTIPLE LINEAR REGRESSION
Module: Introduction to Artificial Intelligence (ST5000CEM)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import pickle
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class HousePricePredictor:
    
    def __init__(self, random_state=42):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.original_features = None
        self.categorical_features = {}
        self.numeric_features = []
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_scaled = self.X_test_scaled = None
        self.performance_metrics = {}
        self.random_state = random_state
        self.data_original = None  # Store original data for visualizations
        print("✓ HousePricePredictor initialized")
    
    def load_data(self, filepath):
        """Load dataset from CSV"""
        try:
            data = pd.read_csv(filepath)
            self.data_original = data.copy()
            print(f"✓ Dataset loaded: {data.shape[0]} rows × {data.shape[1]} columns")
            return data
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return None
    
    def explore_data(self, data):
        print("\n" + "="*70)
        print("DATA EXPLORATION REPORT")
        print("="*70)
        print(f"\nDataset Shape: {data.shape}")
        print(f"\nColumn Names and Types:\n{data.dtypes}")
        print(f"\nMissing Values:\n{data.isnull().sum()}")
        print(f"\nBasic Statistics:\n{data.describe()}")
    
    def preprocess_data(self, data, target_column='price'):
        print("\n" + "="*70)
        print("DATA PREPROCESSING STEPS")
        print("="*70)
        
        data = data.copy()
        
        # Step 1: Handle Missing Values
        print("\n1. HANDLING MISSING VALUES")
        missing_before = data.isnull().sum().sum()
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
                print(f"   ✓ Filled {col} with median")
        
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
                print(f"   ✓ Filled {col} with mode")
        
        print(f"   ✓ Missing values: {missing_before} → {data.isnull().sum().sum()}")
        
        # Step 2: Remove Duplicates
        print("\n2. REMOVING DUPLICATES")
        dup_before = len(data)
        data = data.drop_duplicates()
        print(f"   ✓ Duplicates removed: {dup_before - len(data)}")
        
        # Step 3: Separate Features and Target
        print("\n3. SEPARATING FEATURES AND TARGET")
        if target_column not in data.columns:
            print(f"   ✗ Target column '{target_column}' not found!")
            return None, None
        
        y = data[target_column]
        X = data.drop(columns=[target_column])
        print(f"   ✓ Target: {target_column}")
        
        # Store original features
        self.original_features = X.columns.tolist()
        
        # Step 4: Identify categorical and numeric features
        print("\n4. IDENTIFYING FEATURE TYPES")
        for col in X.select_dtypes(include=['object']).columns:
            self.categorical_features[col] = X[col].unique().tolist()
            print(f"   ✓ Categorical: {col} = {self.categorical_features[col]}")
        
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        print(f"   ✓ Numeric: {self.numeric_features}")
        
        # Step 5: One-Hot Encode Categorical Variables
        print("\n5. CATEGORICAL ENCODING (One-Hot)")
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            print(f"   ✓ Encoded: {cat_cols}")
        
        # Step 6: Remove Outliers
        print("\n6. OUTLIER REMOVAL (IQR Method)")
        Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
        IQR = Q3 - Q1
        mask = (y >= Q1 - 1.5*IQR) & (y <= Q3 + 1.5*IQR)
        X, y = X[mask], y[mask]
        print(f"   ✓ Outliers removed: {sum(~mask)}")
        
        self.feature_names = X.columns.tolist()
        print(f"\n   TOTAL FEATURES: {len(self.feature_names)}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        print("\n" + "="*70)
        print("MODEL TRAINING")
        print("="*70)
        
        print("\n1. TRAIN-TEST SPLIT")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        print(f"   Split ratio: 80%-20%")
        
        print("\n2. FEATURE SCALING (StandardScaler)")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("   ✓ Features scaled to mean=0, std=1")
        
        print("\n3. TRAINING LINEAR REGRESSION MODEL")
        self.model = LinearRegression()
        self.model.fit(self.X_train_scaled, self.y_train)
        print("   ✓ Model trained successfully")
        
        print("\n4. MODEL COEFFICIENTS (Top 10)")
        coeff_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        print(coeff_df.head(10).to_string(index=False))
        print(f"\n   Intercept (b₀): {self.model.intercept_:.2f}")
    
    def evaluate_model(self):
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        print("\n1. TRAINING SET METRICS")
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(self.y_train, y_train_pred)
        print(f"   RMSE (Root Mean Squared Error): {train_rmse:.2f}")
        print(f"   MAE (Mean Absolute Error):      {train_mae:.2f}")
        print(f"   R² Score:                        {train_r2:.4f}")
        print(f"   MAPE:                           {train_mape:.4f}")
        
        print("\n2. TEST SET METRICS")
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(self.y_test, y_test_pred)
        print(f"   RMSE (Root Mean Squared Error): {test_rmse:.2f}")
        print(f"   MAE (Mean Absolute Error):      {test_mae:.2f}")
        print(f"   R² Score:                        {test_r2:.4f}")
        print(f"   MAPE:                           {test_mape:.4f}")
        
        print("\n3. CROSS-VALIDATION (5-Fold)")
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, cv=cv, scoring='r2')
        print(f"   Mean R² Score: {cv_scores.mean():.4f}")
        print(f"   Std Deviation: {cv_scores.std():.4f}")
        print(f"   Individual Fold Scores: {[f'{score:.4f}' for score in cv_scores]}")
        
        self.performance_metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def generate_visualizations(self):
        os.makedirs('figures', exist_ok=True)
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
        # FIGURE 1: Feature Coefficients (Top 10)
        print("\n✓ Generating Figure 4.1: Feature Coefficients...")
        coeff_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in coeff_df['Coefficient']]
        bars = plt.barh(coeff_df['Feature'], coeff_df['Coefficient'], color=colors, edgecolor='black', linewidth=1.2)
        plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title('Top 10 Feature Coefficients - Impact on House Price', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:,.0f}',
                    ha='left' if width > 0 else 'right', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('figures/Figure_4_1_Coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: figures/Figure_4_1_Coefficients.png")
        
        # FIGURE 2: Actual vs Predicted Prices
        print("✓ Generating Figure 4.2: Actual vs Predicted...")
        y_pred = self.model.predict(self.X_test_scaled)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.6, s=100, color='steelblue', edgecolors='black', linewidth=0.5)
        
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')
        
        plt.xlabel('Actual Price', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Price', fontsize=12, fontweight='bold')
        plt.title('Actual vs Predicted House Prices (Test Set)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add R² to plot
        plt.text(0.05, 0.95, f"R² = {self.performance_metrics['test_r2']:.4f}", 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('figures/Figure_4_2_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: figures/Figure_4_2_Actual_vs_Predicted.png")
        
        # FIGURE 3: Residuals Distribution and Plot
        print("✓ Generating Figure 4.3: Residuals Analysis...")
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of residuals
        axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=0, color='red', linestyle='--', lw=2.5)
        axes[0].set_xlabel('Residuals', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Residuals vs Predicted values
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=100, color='steelblue', edgecolors='black', linewidth=0.5)
        axes[1].axhline(y=0, color='red', linestyle='--', lw=2.5)
        axes[1].set_xlabel('Predicted Price', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[1].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('figures/Figure_4_3_Residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: figures/Figure_4_3_Residuals.png")
        
        # FIGURE 4: Performance Metrics Comparison
        print("✓ Generating Figure 4.4: Performance Summary...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['RMSE', 'MAE', 'R²', 'MAPE']
        train_vals = [
            self.performance_metrics['train_rmse'],
            self.performance_metrics['train_mae'],
            self.performance_metrics['train_r2'],
            self.performance_metrics['train_mape']
        ]
        test_vals = [
            self.performance_metrics['test_rmse'],
            self.performance_metrics['test_mae'],
            self.performance_metrics['test_r2'],
            self.performance_metrics['test_mape']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_vals, width, label='Training Set', color='#3498db', edgecolor='black')
        bars2 = ax.bar(x + width/2, test_vals, width, label='Test Set', color='#e74c3c', edgecolor='black')
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('figures/Figure_4_4_Performance_Summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: figures/Figure_4_4_Performance_Summary.png")
        
        # FIGURE 5: Data Distribution
        print("✓ Generating Figure 3.1: Data Distribution...")
        if self.data_original is not None and 'price' in self.data_original.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Price distribution
            axes[0].hist(self.data_original['price'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0].set_xlabel('Price', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[0].set_title('Distribution of House Prices', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3, linestyle='--')
            
            # Area distribution (if available)
            if 'area' in self.data_original.columns:
                axes[1].hist(self.data_original['area'], bins=30, edgecolor='black', alpha=0.7, color='coral')
                axes[1].set_xlabel('Area (sq ft)', fontsize=11, fontweight='bold')
                axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
                axes[1].set_title('Distribution of Property Area', fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            plt.savefig('figures/Figure_3_1_Data_Distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Saved: figures/Figure_3_1_Data_Distribution.png")
        
        # FIGURE 6: Correlation Heatmap
        print("✓ Generating Figure 3.2: Correlation Heatmap...")
        if self.data_original is not None:
            numeric_data = self.data_original.select_dtypes(include=[np.number])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
                        center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('figures/Figure_3_2_Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Saved: figures/Figure_3_2_Correlation_Heatmap.png")
        
        print("\n✓ All visualizations generated successfully!")
    
    def display_figures(self):
        """Display all generated figures info"""
        figure_files = [
            'Figure_3_1_Data_Distribution.png',
            'Figure_3_2_Correlation_Heatmap.png',
            'Figure_4_1_Coefficients.png',
            'Figure_4_2_Actual_vs_Predicted.png',
            'Figure_4_3_Residuals.png',
            'Figure_4_4_Performance_Summary.png'
        ]
        
        print("\n" + "="*70)
        print("AVAILABLE FIGURES")
        print("="*70)
        
        for idx, fig_file in enumerate(figure_files, 1):
            fig_path = f'figures/{fig_file}'
            if os.path.exists(fig_path):
                size_kb = os.path.getsize(fig_path) / 1024
                print(f"\n✓ Figure {idx}: {fig_file}")
                print(f"  Location: {fig_path}")
                print(f"  Size: {size_kb:.2f} KB")
            else:
                print(f"\n✗ Figure {idx}: {fig_file} (Not generated yet)")
    
    def analyze_figure(self, figure_path):
        """Analyze and interpret a generated figure"""
        print("\n" + "="*70)
        print("FIGURE ANALYSIS AND INTERPRETATION")
        print("="*70)
        
        if not os.path.exists(figure_path):
            print(f"✗ Figure not found: {figure_path}")
            return
        
        try:
            img = Image.open(figure_path)
            width, height = img.size
            
            print(f"\n✓ Figure Loaded: {os.path.basename(figure_path)}")
            print(f"  Dimensions: {width} × {height} pixels")
            print(f"  Format: {img.format}")
            print(f"  Size: {os.path.getsize(figure_path) / 1024:.2f} KB")
            
            # Provide interpretation based on figure name
            fig_name = os.path.basename(figure_path).lower()
            
            if 'coefficient' in fig_name:
                print("\n📊 INTERPRETATION:")
                print("  • Shows the top 10 features with greatest impact on house prices")
                print("  • Green bars = features that INCREASE price")
                print("  • Red bars = features that DECREASE price")
                print("  • Longer bars = stronger impact on price")
                if self.performance_metrics:
                    coeff_df = pd.DataFrame({
                        'Feature': self.feature_names,
                        'Coefficient': self.model.coef_
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    print("\n  Key Insights:")
                    for idx, (feat, coef) in enumerate(zip(coeff_df['Feature'].head(3), coeff_df['Coefficient'].head(3)), 1):
                        print(f"    {idx}. {feat}: ${coef:,.2f} impact per unit change")
            
            elif 'actual' in fig_name and 'predicted' in fig_name:
                print("\n📊 INTERPRETATION:")
                print("  • Shows how well the model predicts house prices")
                print("  • Points close to red line = accurate predictions")
                print("  • Points far from red line = inaccurate predictions")
                print("  • R² value shows what % of price variance is explained")
                if self.performance_metrics:
                    r2 = self.performance_metrics['test_r2']
                    rmse = self.performance_metrics['test_rmse']
                    print(f"\n  Model Performance:")
                    print(f"    • R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
                    print(f"    • Average Error: ${rmse:,.2f}")
                    print(f"    • Model is {'EXCELLENT' if r2 > 0.8 else 'GOOD' if r2 > 0.6 else 'FAIR'}")
            
            elif 'residual' in fig_name:
                print("\n📊 INTERPRETATION:")
                print("  • Left plot: Distribution of prediction errors (residuals)")
                print("  • Right plot: Residuals vs predicted values")
                print("  • Ideally, residuals should be centered around zero (red line)")
                print("  • Random scatter = good model; patterns = model issues")
            
            elif 'performance' in fig_name or 'summary' in fig_name:
                print("\n📊 INTERPRETATION:")
                print("  • Compares training vs test performance")
                print("  • Similar heights = good generalization (no overfitting)")
                print("  • Different heights = potential overfitting")
                if self.performance_metrics:
                    print(f"\n  Metrics Explanation:")
                    print(f"    • RMSE: Average error in price units")
                    print(f"    • MAE: Mean Absolute Error")
                    print(f"    • R²: Percentage of variance explained")
                    print(f"    • MAPE: Mean Absolute Percentage Error")
            
            elif 'distribution' in fig_name:
                print("\n📊 INTERPRETATION:")
                print("  • Left: Shows spread of house prices in dataset")
                print("  • Right: Shows distribution of property sizes")
                print("  • Shape indicates data characteristics")
                print("  • Symmetric = normal distribution; skewed = outliers present")
            
            elif 'correlation' in fig_name or 'heatmap' in fig_name:
                print("\n📊 INTERPRETATION:")
                print("  • Shows relationships between numerical features")
                print("  • Red colors = positive correlation")
                print("  • Blue colors = negative correlation")
                print("  • Darker colors = stronger relationships")
                print("  • Numbers show correlation coefficients (-1 to 1)")
            
            print("\n✓ Figure analysis complete!")
            
        except Exception as e:
            print(f"✗ Error analyzing figure: {e}")
    
    def predict(self, features_dict):
        """Make prediction using original feature names"""
        if self.model is None:
            print("✗ Model not trained!")
            return None
        
        try:
            input_data = {}
            
            for feature in self.numeric_features:
                input_data[feature] = features_dict.get(feature, 0)
            
            for feature in self.categorical_features.keys():
                input_data[feature] = features_dict.get(feature, self.categorical_features[feature][0])
            
            features_df = pd.DataFrame([input_data])
            cat_cols = list(self.categorical_features.keys())
            
            if cat_cols:
                features_df = pd.get_dummies(features_df, columns=cat_cols, drop_first=True)
            
            for feature in self.feature_names:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            features_df = features_df[self.feature_names]
            features_scaled = self.scaler.transform(features_df)
            prediction = self.model.predict(features_scaled)[0]
            return prediction
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return None
    
    def get_prediction_input(self):
        """Get user input using ORIGINAL column names"""
        print("\n" + "="*70)
        print("HOUSE PRICE PREDICTION")
        print("="*70)
        print("\nEnter feature values:\n")
        
        features_dict = {}
        
        print("NUMERIC FEATURES:")
        for feature in self.numeric_features:
            try:
                value = float(input(f"  {feature}: "))
                features_dict[feature] = value
            except ValueError:
                print(f"     Invalid input. Using 0.")
                features_dict[feature] = 0
        
        print("\nCATEGORICAL FEATURES:")
        for feature in self.categorical_features.keys():
            options = self.categorical_features[feature]
            print(f"  {feature} options: {', '.join(options)}")
            value = input(f"  {feature}: ").strip()
            
            if value not in options:
                print(f"     Invalid. Using '{options[0]}'")
                value = options[0]
            
            features_dict[feature] = value
        
        return features_dict
    
    def save_model(self, filename='house_model.pkl'):
        """Save trained model"""
        try:
            os.makedirs('models', exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'original_features': self.original_features,
                'categorical_features': self.categorical_features,
                'numeric_features': self.numeric_features,
                'metrics': self.performance_metrics
            }
            with open(f'models/{filename}', 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✓ Model saved: models/{filename}")
            return True
        except Exception as e:
            print(f"✗ Save error: {e}")
            return False
    
    def generate_report(self, filename='house_price_report.txt'):
        """Generate comprehensive summary report"""
        try:
            os.makedirs('outputs', exist_ok=True)
            with open(filename, 'w') as f:
                f.write("="*70 + "\n")
                f.write("HOUSE PRICE PREDICTION MODEL - DETAILED REPORT\n")
                f.write("Algorithm: Multiple Linear Regression\n")
                f.write("="*70 + "\n\n")
                
                f.write("DATASET INFORMATION:\n")
                f.write(f"Training samples: {self.X_train.shape[0]}\n")
                f.write(f"Test samples: {self.X_test.shape[0]}\n")
                f.write(f"Total features (after encoding): {len(self.feature_names)}\n\n")
                
                f.write("ORIGINAL FEATURES:\n")
                f.write(f"Numeric: {self.numeric_features}\n")
                f.write(f"Categorical: {list(self.categorical_features.keys())}\n\n")
                
                f.write("ENCODED FEATURES (Total: {}):\n".format(len(self.feature_names)))
                for feat in self.feature_names:
                    f.write(f"  {feat}\n")
                f.write("\n")
                
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"{'Metric':<30} {'Value':<20}\n")
                f.write("-"*50 + "\n")
                for metric, value in self.performance_metrics.items():
                    f.write(f"{metric:<30} {value:<20.4f}\n")
                f.write("\n")
                
                f.write("TOP 10 COEFFICIENTS:\n")
                f.write(f"{'Feature':<40} {'Coefficient':<20}\n")
                f.write("-"*60 + "\n")
                coeff_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Coefficient': self.model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                for idx, row in coeff_df.head(10).iterrows():
                    f.write(f"{row['Feature']:<40} {row['Coefficient']:<20.6f}\n")
                f.write(f"\nIntercept (b₀): {self.model.intercept_:.2f}\n\n")
                
                f.write("MODEL INTERPRETATION:\n")
                f.write(f"- Model explains {self.performance_metrics['test_r2']*100:.2f}% of price variance\n")
                f.write(f"- Average prediction error: ±${self.performance_metrics['test_rmse']:,.2f}\n")
                f.write(f"- Model generalization (CV R²): {self.performance_metrics['cv_mean']:.4f}\n\n")
                
                f.write("GENERATED FIGURES:\n")
                f.write("1. Figure 3.1 - Data Distribution (Price and Area histograms)\n")
                f.write("2. Figure 3.2 - Correlation Heatmap (Feature relationships)\n")
                f.write("3. Figure 4.1 - Feature Coefficients (Top 10 features)\n")
                f.write("4. Figure 4.2 - Actual vs Predicted (Model accuracy)\n")
                f.write("5. Figure 4.3 - Residuals Analysis (Error distribution)\n")
                f.write("6. Figure 4.4 - Performance Summary (Metrics comparison)\n")
            
            print(f"✓ Report saved: {filename}")
            return True
        except Exception as e:
            print(f"✗ Report error: {e}")
            return False


def main():
    """Main CLI Interface with Enhanced Features"""
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION SYSTEM")
    print("Multiple Linear Regression Implementation")
    print("="*70)
    
    predictor = HousePricePredictor()
    
    while True:
        print("\n--- MAIN MENU ---")
        print("1. Load and train model")
        print("2. Make prediction")
        print("3. View metrics")
        print("4. Generate visualizations")
        print("5. View generated figures")
        print("6. Analyze figure")
        print("7. Save model")
        print("8. Generate report")
        print("9. Exit")
        
        choice = input("\nSelect (1-9): ").strip()
        
        if choice == '1':
            filepath = input("CSV path: ").strip()
            data = predictor.load_data(filepath)
            if data is not None:
                predictor.explore_data(data)
                target = input("Target column (default 'price'): ").strip() or 'price'
                X, y = predictor.preprocess_data(data, target)
                if X is not None:
                    predictor.train_model(X, y)
                    predictor.evaluate_model()
                    print("\n✓ Model training complete!")
        
        elif choice == '2':
            if predictor.model is None:
                print("✗ Train model first (option 1)")
            else:
                features = predictor.get_prediction_input()
                pred = predictor.predict(features)
                if pred:
                    print(f"\n{'='*70}")
                    print(f"PREDICTED HOUSE PRICE: ${pred:,.2f}")
                    print(f"{'='*70}\n")
        
        elif choice == '3':
            if predictor.performance_metrics:
                print("\n--- MODEL PERFORMANCE METRICS ---")
                for m, v in predictor.performance_metrics.items():
                    print(f"{m:20s}: {v:.4f}")
            else:
                print("✗ No metrics. Train model first.")
        
        elif choice == '4':
            if predictor.model is None:
                print("✗ Train model first (option 1)")
            else:
                predictor.generate_visualizations()
        
        elif choice == '5':
            if predictor.model is None:
                print("✗ Train model first (option 1)")
            else:
                predictor.display_figures()
        
        elif choice == '6':
            figure_name = input("Enter figure filename (e.g., figures/Figure_4_1_Coefficients.png): ").strip()
            predictor.analyze_figure(figure_name)
        
        elif choice == '7':
            name = input("Filename (default 'house_model.pkl'): ").strip()
            predictor.save_model(name or 'house_model.pkl')
        
        elif choice == '8':
            name = input("Report name (default 'house_price_report.txt'): ").strip()
            predictor.generate_report(name or 'house_price_report.txt')
        
        elif choice == '9':
            print("\nGoodbye!")
            break
        
        else:
            print("✗ Invalid option. Please try again.")


if __name__ == "__main__":
    main()