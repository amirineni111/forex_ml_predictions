"""
Database Export Module for Forex Predictions

This module handles writing prediction results back to SQL Server database.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy import text
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import ForexSQLServerConnection

logger = logging.getLogger(__name__)


class ForexResultsExporter:
    """Handles exporting forex prediction results to SQL Server."""
    
    def __init__(self, connection: Optional[ForexSQLServerConnection] = None):
        """Initialize the results exporter."""
        self.db = connection or ForexSQLServerConnection()
        
        # Default table names for results
        self.predictions_table = 'forex_ml_predictions'
        self.model_performance_table = 'forex_model_performance'
        self.daily_summary_table = 'forex_daily_summary'
    
    def create_results_tables(self):
        """Create the results tables if they don't exist."""
        
        predictions_table_sql = f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{self.predictions_table}' AND xtype='U')
        CREATE TABLE {self.predictions_table} (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            prediction_date DATETIME NOT NULL,
            currency_pair VARCHAR(10) NOT NULL,
            date_time DATETIME NOT NULL,
            open_price DECIMAL(10,5),
            high_price DECIMAL(10,5),
            low_price DECIMAL(10,5),
            close_price DECIMAL(10,5),
            volume DECIMAL(15,2),
            predicted_signal VARCHAR(10) NOT NULL,
            signal_confidence DECIMAL(5,3),
            prob_buy DECIMAL(5,3),
            prob_sell DECIMAL(5,3),
            prob_hold DECIMAL(5,3),
            model_name VARCHAR(50),
            model_version VARCHAR(20),
            features_used INT,
            created_at DATETIME DEFAULT GETDATE(),
            INDEX IX_forex_predictions_pair_date (currency_pair, date_time),
            INDEX IX_forex_predictions_signal (predicted_signal),
            INDEX IX_forex_predictions_created (created_at)
        )
        """
        
        performance_table_sql = f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{self.model_performance_table}' AND xtype='U')
        CREATE TABLE {self.model_performance_table} (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            model_name VARCHAR(50) NOT NULL,
            currency_pair VARCHAR(10),
            training_date DATETIME NOT NULL,
            cv_accuracy_mean DECIMAL(5,3),
            cv_accuracy_std DECIMAL(5,3),
            train_accuracy DECIMAL(5,3),
            train_precision DECIMAL(5,3),
            train_recall DECIMAL(5,3),
            train_f1 DECIMAL(5,3),
            train_auc DECIMAL(5,3),
            training_samples INT,
            features_count INT,
            signal_type VARCHAR(20),
            model_params TEXT,
            created_at DATETIME DEFAULT GETDATE(),
            INDEX IX_model_performance_name (model_name),
            INDEX IX_model_performance_pair (currency_pair),
            INDEX IX_model_performance_date (training_date)
        )
        """
        
        # Feature values table for storing all 28 technical indicators
        features_table_sql = f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='forex_prediction_features' AND xtype='U')
        CREATE TABLE forex_prediction_features (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            prediction_date DATETIME NOT NULL,
            currency_pair VARCHAR(10) NOT NULL,
            -- Price features
            open_price DECIMAL(10,5),
            high_price DECIMAL(10,5), 
            low_price DECIMAL(10,5),
            close_price DECIMAL(10,5),
            volume DECIMAL(15,2),
            -- Moving averages
            sma_5 DECIMAL(10,5),
            sma_10 DECIMAL(10,5),
            sma_20 DECIMAL(10,5),
            sma_50 DECIMAL(10,5),
            sma_200 DECIMAL(10,5),
            ema_5 DECIMAL(10,5),
            ema_10 DECIMAL(10,5),
            ema_20 DECIMAL(10,5),
            ema_50 DECIMAL(10,5),
            ema_200 DECIMAL(10,5),
            -- Technical indicators
            rsi DECIMAL(8,4),
            macd DECIMAL(10,6),
            macd_signal DECIMAL(10,6),
            macd_histogram DECIMAL(10,6),
            atr DECIMAL(10,6),
            -- Bollinger bands
            bb_upper DECIMAL(10,5),
            bb_middle DECIMAL(10,5),
            bb_lower DECIMAL(10,5),
            bb_width DECIMAL(10,6),
            bb_percent DECIMAL(8,4),
            -- Other features
            daily_return DECIMAL(10,6),
            gap DECIMAL(10,6),
            volume_ratio DECIMAL(8,4),
            -- Prediction info
            predicted_signal VARCHAR(10),
            signal_confidence DECIMAL(5,3),
            model_name VARCHAR(50),
            created_at DATETIME DEFAULT GETDATE(),
            INDEX IX_features_pair_date (currency_pair, prediction_date),
            INDEX IX_features_signal (predicted_signal)
        )
        """
        
        summary_table_sql = f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{self.daily_summary_table}' AND xtype='U')
        CREATE TABLE {self.daily_summary_table} (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            summary_date DATE NOT NULL,
            currency_pair VARCHAR(10) NOT NULL,
            total_predictions INT,
            buy_signals INT,
            sell_signals INT,
            hold_signals INT,
            avg_confidence DECIMAL(5,3),
            high_confidence_signals INT,
            model_used VARCHAR(50),
            created_at DATETIME DEFAULT GETDATE(),
            UNIQUE (summary_date, currency_pair),
            INDEX IX_daily_summary_date (summary_date),
            INDEX IX_daily_summary_pair (currency_pair)
        )
        """
        
        try:
            engine = self.db.get_sqlalchemy_engine()
            
            with engine.connect() as conn:
                # Create predictions table
                conn.execute(text(predictions_table_sql))
                conn.commit()
                logger.info(f"✅ Created/verified {self.predictions_table} table")
                
                # Create performance table
                conn.execute(text(performance_table_sql))
                conn.commit()
                logger.info(f"✅ Created/verified {self.model_performance_table} table")
                
                # Create summary table
                conn.execute(text(summary_table_sql))
                conn.commit()
                logger.info(f"✅ Created/verified {self.daily_summary_table} table")
                
                # Create features table
                conn.execute(text(features_table_sql))
                conn.commit()
                logger.info(f"✅ Created/verified forex_prediction_features table")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating results tables: {e}")
            return False

    def drop_prediction_tables(self) -> bool:
        """
        Drop existing forex prediction tables to start fresh.
        
        Returns:
            bool: True if successful, False otherwise
        """
        drop_tables_sql = f"""
        -- Drop prediction tables if they exist
        IF EXISTS (SELECT * FROM sysobjects WHERE name='{self.predictions_table}' AND xtype='U')
            DROP TABLE {self.predictions_table}
        
        IF EXISTS (SELECT * FROM sysobjects WHERE name='{self.daily_summary_table}' AND xtype='U')
            DROP TABLE {self.daily_summary_table}
        
        IF EXISTS (SELECT * FROM sysobjects WHERE name='{self.model_performance_table}' AND xtype='U')
            DROP TABLE {self.model_performance_table}
        """
        
        try:
            engine = self.db.get_sqlalchemy_engine()
            
            with engine.connect() as conn:
                conn.execute(text(drop_tables_sql))
                conn.commit()
                logger.info(f"✅ Dropped existing forex prediction tables")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Error dropping tables: {e}")
            return False

    def recreate_prediction_tables(self) -> bool:
        """
        Drop and recreate all forex prediction tables with fresh schema.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Drop existing tables
            if not self.drop_prediction_tables():
                return False
                
            # Recreate tables
            if not self.create_results_tables():
                return False
                
            logger.info("✅ Successfully recreated all forex prediction tables")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recreating tables: {e}")
            return False
    
    def export_predictions(self, predictions_df: pd.DataFrame, model_name: str = 'forex_ml_model', 
                          model_version: str = '1.0') -> bool:
        """
        Export prediction results to SQL Server.
        
        Args:
            predictions_df: DataFrame with prediction results
            model_name: Name of the model used for predictions
            model_version: Version of the model
            
        Returns:
            True if successful, False otherwise
        """
        
        if predictions_df.empty:
            logger.warning("No predictions to export")
            return False
        
        try:
            # Prepare data for export
            export_df = predictions_df.copy()
            
            # Add metadata
            export_df['prediction_date'] = datetime.now()
            export_df['model_name'] = model_name
            export_df['model_version'] = model_version
            export_df['features_used'] = len([col for col in predictions_df.columns 
                                            if col not in ['signal', 'prob_BUY', 'prob_SELL', 'prob_HOLD', 
                                                          'confidence', 'date_time', 'currency_pair']])
            
            # Rename columns to match database schema
            column_mapping = {
                'signal': 'predicted_signal',
                'confidence': 'signal_confidence',
                'prob_BUY': 'prob_buy',
                'prob_SELL': 'prob_sell', 
                'prob_HOLD': 'prob_hold'
            }
            export_df = export_df.rename(columns=column_mapping)
            
            # Select only columns that exist in the table
            required_columns = [
                'prediction_date', 'currency_pair', 'date_time', 'open_price', 
                'high_price', 'low_price', 'close_price', 'volume',
                'predicted_signal', 'signal_confidence', 'prob_buy', 'prob_sell', 
                'prob_hold', 'model_name', 'model_version', 'features_used'
            ]
            
            # Filter to only include available columns
            available_columns = [col for col in required_columns if col in export_df.columns]
            missing_columns = [col for col in required_columns if col not in export_df.columns]
            
            if missing_columns:
                logger.info(f"Missing columns detected: {missing_columns}")
                # Add missing columns with appropriate defaults
                for col in missing_columns:
                    if col in ['volume']:
                        export_df[col] = 0
                    elif col in ['open_price', 'high_price', 'low_price', 'close_price']:
                        export_df[col] = 0.0
                    elif col in ['signal_confidence']:
                        export_df[col] = 0.5
                    elif col in ['prob_buy', 'prob_sell', 'prob_hold']:
                        export_df[col] = 0.0
                    else:
                        export_df[col] = None
            
            # Ensure all required columns are present and in the right order
            export_df = export_df[required_columns]
            
            # Handle missing optional columns
            if 'signal_confidence' not in export_df.columns:
                export_df['signal_confidence'] = 0.5  # Default confidence
            
            if 'prob_buy' not in export_df.columns:
                export_df['prob_buy'] = None
            if 'prob_sell' not in export_df.columns:
                export_df['prob_sell'] = None
            if 'prob_hold' not in export_df.columns:
                export_df['prob_hold'] = None
            
            # Clean data
            export_df = export_df.replace([np.inf, -np.inf], np.nan)
            
            # Export to SQL Server using raw SQL instead of pandas to_sql
            engine = self.db.get_sqlalchemy_engine()
            
            # Define the columns we want to insert (excluding auto-generated columns like 'id' and 'created_at')
            columns_to_insert = ['prediction_date', 'currency_pair', 'date_time', 
                               'open_price', 'high_price', 'low_price', 'close_price', 
                               'volume', 'predicted_signal', 'signal_confidence', 
                               'prob_buy', 'prob_sell', 'prob_hold', 'model_name', 
                               'model_version', 'features_used']
            
            # Ensure we only include columns that exist in the dataframe
            available_columns = [col for col in columns_to_insert if col in export_df.columns]
            final_df = export_df[available_columns].copy()
            
            logger.info(f"Inserting {len(final_df)} rows with columns: {list(final_df.columns)}")
            
            # Use raw SQL insert with named parameters instead of pandas to_sql to avoid parameter issues
            insert_sql = f"""
                INSERT INTO {self.predictions_table} 
                ({', '.join(available_columns)})
                VALUES ({', '.join([':' + col for col in available_columns])})
            """
            
            with engine.connect() as conn:
                rows_inserted = 0
                for _, row in final_df.iterrows():
                    try:
                        # Convert row to dictionary format for execute
                        param_dict = {col: row[col] for col in available_columns}
                        conn.execute(text(insert_sql), param_dict)
                        rows_inserted += 1
                    except Exception as row_error:
                        logger.warning(f"Failed to insert row: {row_error}")
                        continue
                conn.commit()
                
            logger.info(f"Successfully inserted {rows_inserted} out of {len(final_df)} rows")
            
            logger.info(f"✅ Exported {len(export_df)} prediction records to {self.predictions_table}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting daily summary: {e}")
            return False
    
    def export_prediction_features(self, features_df: pd.DataFrame, predictions_df: pd.DataFrame) -> bool:
        """
        Export prediction feature values to the features table for analysis.
        
        Args:
            features_df: DataFrame with all feature values used for predictions
            predictions_df: DataFrame with prediction results
            
        Returns:
            True if successful, False otherwise
        """
        if features_df.empty or predictions_df.empty:
            logger.warning("No feature data to export")
            return False
            
        try:
            # Merge feature values with prediction results
            combined_df = features_df.merge(
                predictions_df[['currency_pair', 'predicted_signal', 'signal_confidence', 'model_name']], 
                on='currency_pair', 
                how='left'
            )
            
            # Add prediction_date from today
            from datetime import datetime, timedelta
            tomorrow = datetime.now().date() + timedelta(days=1)
            combined_df['prediction_date'] = tomorrow
            
            # Reorder columns to match table schema
            feature_columns = [
                'prediction_date', 'currency_pair',
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
                'rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
                'daily_return', 'gap', 'volume_ratio',
                'predicted_signal', 'signal_confidence', 'model_name'
            ]
            
            # Select only columns that exist in the dataframe
            available_columns = [col for col in feature_columns if col in combined_df.columns]
            export_df = combined_df[available_columns].copy()
            
            # Fill missing values
            export_df = export_df.fillna(0)
            
            logger.info(f"Inserting {len(export_df)} feature records with columns: {list(export_df.columns)}")
            
            engine = self.db.get_sqlalchemy_engine()
            rows_inserted = export_df.to_sql(
                'forex_prediction_features',
                engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info(f"✅ Exported {len(export_df)} feature records to forex_prediction_features")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting prediction features: {e}")
            return False
    
    def export_model_performance(self, model_results: Dict = None, model_name: str = 'forex_ml_model',
                               training_pairs: List[str] = None, training_date: datetime = None) -> bool:
        """
        Export model performance metrics to SQL Server for weekly retraining.
        
        Args:
            model_results: Dictionary of model results from training
            model_name: Name of the model used
            training_pairs: List of currency pairs used for training
            training_date: Date when training occurred
            
        Returns:
            True if successful, False otherwise
        """
        
        if model_results is None or not model_results:
            logger.warning("No model results to export")
            return False
        
        try:
            performance_records = []
            training_date = training_date or datetime.now()
            currency_pairs_str = ','.join(training_pairs) if training_pairs else 'ALL'
            
            for model_type, metrics in model_results.items():
                if isinstance(metrics, dict) and 'cv_accuracy_mean' in metrics:
                    record = {
                        'model_name': f"{model_name}_{model_type}",
                        'currency_pair': currency_pairs_str,
                        'training_date': training_date,
                        'cv_accuracy_mean': metrics.get('cv_accuracy_mean', 0.0),
                        'cv_accuracy_std': metrics.get('cv_accuracy_std', 0.0),
                        'train_accuracy': metrics.get('train_accuracy', 0.0),
                        'train_precision': metrics.get('train_precision', 0.0),
                        'train_recall': metrics.get('train_recall', 0.0),
                        'train_f1': metrics.get('train_f1', 0.0),
                        'train_auc': metrics.get('train_auc', 0.0),
                        'training_samples': metrics.get('training_samples', 0),
                        'features_count': metrics.get('features_count', 0),
                        'signal_type': 'multi_currency',
                        'model_params': str(metrics)
                    }
                    performance_records.append(record)
            
            if performance_records:
                performance_df = pd.DataFrame(performance_records)
                
                # Export to SQL Server using raw SQL
                engine = self.db.get_sqlalchemy_engine()
                
                insert_sql = f"""
                    INSERT INTO {self.model_performance_table} 
                    (model_name, currency_pair, training_date, cv_accuracy_mean, cv_accuracy_std,
                     train_accuracy, train_precision, train_recall, train_f1, train_auc,
                     training_samples, features_count, signal_type, model_params)
                    VALUES (:model_name, :currency_pair, :training_date, :cv_accuracy_mean, :cv_accuracy_std,
                            :train_accuracy, :train_precision, :train_recall, :train_f1, :train_auc,
                            :training_samples, :features_count, :signal_type, :model_params)
                """
                
                with engine.connect() as conn:
                    for _, row in performance_df.iterrows():
                        param_dict = row.to_dict()
                        conn.execute(text(insert_sql), param_dict)
                    conn.commit()
                
                logger.info(f"✅ Exported {len(performance_records)} model performance records")
                return True
            else:
                logger.warning("No valid model results to export")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error exporting model performance: {e}")
            return False
    
    def export_prediction_features(self, features_df: pd.DataFrame, predictions_df: pd.DataFrame) -> bool:
        """
        Export prediction feature values to the features table for analysis.
        
        Args:
            features_df: DataFrame with all feature values used for predictions
            predictions_df: DataFrame with prediction results
            
        Returns:
            True if successful, False otherwise
        """
        if features_df.empty or predictions_df.empty:
            logger.warning("No feature data to export")
            return False
            
        try:
            # Merge feature values with prediction results
            combined_df = features_df.merge(
                predictions_df[['currency_pair', 'predicted_signal', 'signal_confidence', 'model_name']], 
                on='currency_pair', 
                how='left'
            )
            
            # Add prediction_date from tomorrow
            from datetime import datetime, timedelta
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            # Skip weekends
            while tomorrow.weekday() >= 5:
                tomorrow = tomorrow + timedelta(days=1)
            combined_df['prediction_date'] = tomorrow
            
            # Define all feature columns in correct order
            feature_columns = [
                'prediction_date', 'currency_pair',
                'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
                'rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
                'daily_return', 'gap', 'volume_ratio',
                'predicted_signal', 'signal_confidence', 'model_name'
            ]
            
            # Select only columns that exist in the dataframe
            available_columns = [col for col in feature_columns if col in combined_df.columns]
            export_df = combined_df[available_columns].copy()
            
            # Fill missing values with 0
            export_df = export_df.fillna(0)
            
            logger.info(f"Inserting {len(export_df)} feature records with {len(available_columns)} columns")
            
            engine = self.db.get_sqlalchemy_engine()
            rows_inserted = export_df.to_sql(
                'forex_prediction_features',
                engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info(f"✅ Exported {len(export_df)} feature records to forex_prediction_features")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting prediction features: {e}")
            return False
    
    def export_daily_summary(self, predictions_df: pd.DataFrame, model_name: str = 'forex_ml_model') -> bool:
        """
        Export daily summary statistics to SQL Server.
        
        Args:
            predictions_df: DataFrame with prediction results
            model_name: Name of the model used
            
        Returns:
            True if successful, False otherwise
        """
        
        if predictions_df.empty:
            logger.warning("No predictions for daily summary")
            return False
        
        try:
            # Create daily summary by currency pair
            summary_data = []
            
            # Get today's date
            today = datetime.now().date()
            
            # Group by currency pair
            for currency_pair in predictions_df['currency_pair'].unique():
                pair_data = predictions_df[predictions_df['currency_pair'] == currency_pair]
                
                # Calculate summary statistics
                total_predictions = len(pair_data)
                buy_signals = len(pair_data[pair_data.get('signal', pair_data.get('predicted_signal')) == 'BUY'])
                sell_signals = len(pair_data[pair_data.get('signal', pair_data.get('predicted_signal')) == 'SELL'])
                hold_signals = len(pair_data[pair_data.get('signal', pair_data.get('predicted_signal')) == 'HOLD'])
                
                # Calculate confidence statistics
                confidence_col = 'confidence' if 'confidence' in pair_data.columns else 'signal_confidence'
                if confidence_col in pair_data.columns:
                    avg_confidence = pair_data[confidence_col].mean()
                    high_confidence_signals = len(pair_data[pair_data[confidence_col] > 0.7])
                else:
                    avg_confidence = 0.5
                    high_confidence_signals = 0
                
                summary_record = {
                    'summary_date': today,
                    'currency_pair': currency_pair,
                    'total_predictions': total_predictions,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'hold_signals': hold_signals,
                    'avg_confidence': avg_confidence,
                    'high_confidence_signals': high_confidence_signals,
                    'model_used': model_name
                }
                
                summary_data.append(summary_record)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # Use MERGE to insert/update daily summary
                engine = self.db.get_sqlalchemy_engine()
                
                with engine.connect() as conn:
                    # Delete existing records for today
                    delete_sql = text(f"""
                        DELETE FROM {self.daily_summary_table} 
                        WHERE summary_date = :today
                    """)
                    conn.execute(delete_sql, {'today': today})
                    
                    # Insert new records
                    summary_df.to_sql(
                        name=self.daily_summary_table,
                        con=conn,
                        if_exists='append',
                        index=False,
                        method='multi'
                    )
                    
                    conn.commit()
                
                logger.info(f"✅ Exported daily summary for {len(summary_data)} currency pairs")
                return True
            else:
                logger.warning("No summary data to export")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error exporting daily summary: {e}")
            return False
    
    def get_recent_predictions(self, currency_pair: Optional[str] = None, 
                             days_back: int = 7) -> pd.DataFrame:
        """
        Retrieve recent predictions from the database.
        
        Args:
            currency_pair: Specific currency pair (optional)
            days_back: Number of days to look back
            
        Returns:
            DataFrame with recent predictions
        """
        
        try:
            where_clause = f"WHERE prediction_date >= DATEADD(day, -{days_back}, GETDATE())"
            
            if currency_pair:
                where_clause += f" AND currency_pair = '{currency_pair}'"
            
            query = f"""
            SELECT 
                prediction_date,
                currency_pair,
                date_time,
                close_price,
                predicted_signal,
                signal_confidence,
                prob_buy,
                prob_sell,
                prob_hold,
                model_name
            FROM {self.predictions_table}
            {where_clause}
            ORDER BY currency_pair, date_time DESC
            """
            
            return pd.read_sql(query, self.db.get_sqlalchemy_engine())
            
        except Exception as e:
            logger.error(f"❌ Error retrieving recent predictions: {e}")
            return pd.DataFrame()
    
    def get_model_performance_history(self, model_name: Optional[str] = None,
                                    currency_pair: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve model performance history from the database.
        
        Args:
            model_name: Specific model name (optional)
            currency_pair: Specific currency pair (optional)
            
        Returns:
            DataFrame with performance history
        """
        
        try:
            where_conditions = []
            
            if model_name:
                where_conditions.append(f"model_name = '{model_name}'")
            
            if currency_pair:
                where_conditions.append(f"currency_pair = '{currency_pair}'")
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            query = f"""
            SELECT 
                model_name,
                currency_pair,
                training_date,
                cv_accuracy_mean,
                train_accuracy,
                train_f1,
                training_samples,
                features_count,
                signal_type
            FROM {self.model_performance_table}
            {where_clause}
            ORDER BY training_date DESC
            """
            
            return pd.read_sql(query, self.db.get_sqlalchemy_engine())
            
        except Exception as e:
            logger.error(f"❌ Error retrieving performance history: {e}")
            return pd.DataFrame()


# Convenience functions
def create_results_exporter() -> ForexResultsExporter:
    """Create a forex results exporter instance."""
    return ForexResultsExporter()


def setup_results_tables() -> bool:
    """Setup the results tables in SQL Server."""
    exporter = ForexResultsExporter()
    return exporter.create_results_tables()