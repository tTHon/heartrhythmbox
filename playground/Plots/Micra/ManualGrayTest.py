import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import warnings

# -------------------------------------------------------------------
# 1. CLASS DEFINITION (as provided)
# -------------------------------------------------------------------
class ManualFineGrayFitter:
    """
    Manual implementation of Fine-Gray regression model for competing risks.
    """
    def __init__(self):
        self.coefficients_ = None
        self.covariance_matrix_ = None
        self.log_likelihood_ = None
        self.fitted = False
        self.covariate_names_ = None
        self.n_observations_ = None
        self.n_events_ = None

    def _calculate_weights(self, duration, status):
        """
        Calculate inverse probability of censoring weights (simplified version).
        In practice, this would use Kaplan-Meier for censoring distribution.
        """
        n = len(duration)
        weights = np.ones(n)
        sort_idx = np.argsort(duration)
        sorted_duration = duration[sort_idx]
        sorted_status = status[sort_idx]
        
        for i in range(n):
            if sorted_status[i] == 2:  # Competing event
                at_risk = np.sum(sorted_duration >= sorted_duration[i])
                if at_risk > 1:
                    weights[sort_idx[i]] = at_risk / (at_risk - 1)
        return weights

    def _create_finegray_data(self, df, duration_col, event_col, covariate_cols, event_of_interest):
        """
        Transform data for Fine-Gray model.
        """
        new_rows = []
        unique_event_times = sorted(df[df[event_col] == event_of_interest][duration_col].unique())

        for _, row in df.iterrows():
            obs_time = row[duration_col]
            obs_status = row[event_col]
            
            # Subject had the event of interest
            if obs_status == event_of_interest:
                new_rows.append({**row, 'fg_stop': obs_time, 'fg_event': 1})
            
            # Subject was censored before any event
            elif obs_status == 0:
                new_rows.append({**row, 'fg_stop': obs_time, 'fg_event': 0})

            # Subject had a competing event
            elif obs_status != event_of_interest and obs_status != 0:
                # Add their actual observation time (censored for event of interest)
                new_rows.append({**row, 'fg_stop': obs_time, 'fg_event': 0})
                
                # Add artificial rows for this subject for each future event time
                for t in unique_event_times:
                    if t > obs_time:
                        artificial_row = row.copy()
                        artificial_row['fg_stop'] = t
                        artificial_row['fg_event'] = 0 # Censored
                        artificial_row['artificial'] = True # Mark as artificial
                        new_rows.append(artificial_row)

        fg_df = pd.DataFrame(new_rows)
        return fg_df

    def _log_partial_likelihood(self, beta, X, time, event, weights):
        """
        Log partial likelihood for Cox regression with weights.
        """
        log_likelihood = 0.0
        # Sort by time for risk set calculation
        sort_idx = np.argsort(time)
        X_sorted, time_sorted, event_sorted, weights_sorted = X[sort_idx], time[sort_idx], event[sort_idx], weights[sort_idx]

        for i in range(len(time_sorted)):
            if event_sorted[i] == 1:
                risk_set_indices = np.where(time_sorted >= time_sorted[i])[0]
                if len(risk_set_indices) == 0: continue
                
                risk_set_X = X_sorted[risk_set_indices]
                risk_set_weights = weights_sorted[risk_set_indices]
                
                linear_pred_i = np.dot(X_sorted[i], beta)
                risk_set_linear_pred = np.dot(risk_set_X, beta)
                
                # Log-sum-exp for stability
                max_pred = np.max(risk_set_linear_pred)
                log_sum_exp = max_pred + np.log(np.sum(risk_set_weights * np.exp(risk_set_linear_pred - max_pred)))
                
                log_likelihood += weights_sorted[i] * (linear_pred_i - log_sum_exp)
                
        return -log_likelihood

    def _log_likelihood_gradient(self, beta, X, time, event, weights):
        """
        Gradient of log partial likelihood.
        """
        p = X.shape[1]
        gradient = np.zeros(p)
        sort_idx = np.argsort(time)
        X_sorted, time_sorted, event_sorted, weights_sorted = X[sort_idx], time[sort_idx], event[sort_idx], weights[sort_idx]

        for i in range(len(time_sorted)):
            if event_sorted[i] == 1:
                risk_set_indices = np.where(time_sorted >= time_sorted[i])[0]
                if len(risk_set_indices) == 0: continue
                
                risk_set_X = X_sorted[risk_set_indices]
                risk_set_weights = weights_sorted[risk_set_indices]
                risk_set_linear_pred = np.dot(risk_set_X, beta)

                exp_pred = np.exp(risk_set_linear_pred)
                weighted_exp = risk_set_weights * exp_pred

                weighted_sum_exp = np.sum(weighted_exp)
                if weighted_sum_exp > 0:
                    expected_X = np.sum(risk_set_X * weighted_exp[:, np.newaxis], axis=0) / weighted_sum_exp
                    gradient += weights_sorted[i] * (X_sorted[i] - expected_X)
        return -gradient

    def fit(self, df, duration_col, event_col, formula=None, covariate_cols=None, event_of_interest=1):
        if formula is not None:
            covariate_cols = [c.strip() for c in formula.split('+')]
        elif covariate_cols is None:
            raise ValueError("Must specify either formula or covariate_cols")
            
        self.covariate_names_ = covariate_cols
        
        # Data preparation
        analysis_df = df[[duration_col, event_col] + self.covariate_names_].dropna().copy()
        
        # This implementation requires a more complex data duplication scheme and IPCW.
        # For simplicity in this manual version, we will use a weighted Cox PH on the original data,
        # which is an approximation. A true Fine-Gray model requires data augmentation.
        print("Note: Using a simplified weighted Cox PH as an approximation for the manual model.")
        
        X = analysis_df[self.covariate_names_].values
        time = analysis_df[duration_col].values
        status = analysis_df[event_col].values
        
        # Create event indicator for the cause of interest
        event = (status == event_of_interest).astype(int)
        
        # Calculate simplified IPCW
        weights = self._calculate_weights(time, status)
        
        # Standardize covariates
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_standardized = (X - X_mean) / X_std
        
        initial_beta = np.zeros(X.shape[1])
        
        print("Optimizing log partial likelihood...")
        result = minimize(
            fun=self._log_partial_likelihood,
            x0=initial_beta,
            args=(X_standardized, time, event, weights),
            jac=self._log_likelihood_gradient,
            method='BFGS'
        )

        if result.success:
            self.coefficients_ = result.x / X_std
            self.log_likelihood_ = -result.fun
            # Simplified covariance using inverse Hessian
            self.covariance_matrix_ = result.hess_inv / np.outer(X_std, X_std)
            self.fitted = True
            self.n_observations_ = len(analysis_df)
            self.n_events_ = np.sum(event)
            print("Manual Fine-Gray model fitted successfully!")
        else:
            raise RuntimeError(f"Optimization failed: {result.message}")

    def print_summary(self):
        if not self.fitted:
            print("Model has not been fitted yet.")
            return
        
        print("\n" + "="*80)
        print("Manual Fine-Gray Regression Model Summary")
        print("="*80)
        print(f"Number of observations: {self.n_observations_}")
        print(f"Number of events of interest: {self.n_events_}")
        print(f"Log-likelihood: {self.log_likelihood_:.4f}\n")
        
        std_errors = np.sqrt(np.diag(self.covariance_matrix_))
        z_scores = self.coefficients_ / std_errors
        p_values = 2 * stats.norm.sf(np.abs(z_scores))
        
        hazard_ratios = np.exp(self.coefficients_)
        hr_lower_ci = np.exp(self.coefficients_ - 1.96 * std_errors)
        hr_upper_ci = np.exp(self.coefficients_ + 1.96 * std_errors)
        
        summary_df = pd.DataFrame({
            'coef': self.coefficients_,
            'exp(coef)': hazard_ratios,
            'se(coef)': std_errors,
            'z': z_scores,
            'p': p_values,
            'lower 0.95': hr_lower_ci,
            'upper 0.95': hr_upper_ci
        }, index=self.covariate_names_)
        
        print(summary_df.round(4))
        print("="*80)

# -------------------------------------------------------------------
# 2. DATA LOADING AND PREPROCESSING
# -------------------------------------------------------------------
try:
    df = pd.read_csv('playground/Micra/LP_events.csv')
except FileNotFoundError:
    print("Error: The file 'playground/Micra/LP_events.csv' was not found.")
    exit()

# Preprocessing steps
df.dropna(subset=['T2Events', 'AnyEvents'], inplace=True)
df['T2Events'] = pd.to_numeric(df['T2Events'], errors='coerce')
df['AnyEvents'] = df['AnyEvents'].astype(int)
df['status'] = df['AnyEvents'].apply(lambda x: 1 if x == 1 else 2 if x == 2 else 0)
df['duration'] = df['T2Events']
# Ensure 'Age' and 'lowBSA' are clean for the model
df.dropna(subset=['duration', 'status', 'Age', 'lowBSA'], inplace=True)
print("Data loaded and preprocessed successfully.")
print(f"Total subjects available for modeling: {len(df)}")
print("-" * 50)

# -------------------------------------------------------------------
# 3. IMPLEMENTATION ON YOUR DATA
# -------------------------------------------------------------------
# This script will now use your ManualFineGrayFitter class to analyze
# the effect of different variables on the risk of 'complications' (status=1).

# --- Example 1: Fit model with a continuous variable ('Age') ---
print("\n--- Fitting model for the 'Age' variable ---")
try:
    # 1. Instantiate the model
    manual_model_age = ManualFineGrayFitter()

    # 2. Fit the model on your data
    manual_model_age.fit(df,
                         duration_col='duration',
                         event_col='status',
                         formula='Age',
                         event_of_interest=1)

    # 3. Print the results
    manual_model_age.print_summary()

except Exception as e:
    print(f"\nAn error occurred while fitting the model for 'Age': {e}")
    
# --- Example 2: Fit model with a binary variable ('lowBSA') ---
print("\n--- Fitting model for the 'lowBSA' variable ---")
try:
    # 1. Instantiate the model
    manual_model_bsa = ManualFineGrayFitter()

    # 2. Fit the model on your data
    manual_model_bsa.fit(df,
                         duration_col='duration',
                         event_col='status',
                         formula='lowBSA',
                         event_of_interest=1)

    # 3. Print the results
    manual_model_bsa.print_summary()

except Exception as e:
    print(f"\nAn error occurred while fitting the model for 'lowBSA': {e}")