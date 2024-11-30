import numpy as np
from scipy.stats import norm

def z_test(data, population_mean, population_std, alpha=0.05):
    """
    Conduct a one-sample z-test.
    
    Parameters:
    - data (list or array): The sample data
    - population_mean (float): The known population mean
    - population_std (float): The known population standard deviation
    - alpha (float): The significance level (default 0.05)
    
    Returns:
    - z_score (float): The calculated z-score
    - p_value (float): The calculated p-value
    - decision (str): Decision to "Reject" or "Fail to Reject" the null hypothesis
    """
    # Calculate sample statistics
    sample_mean = np.mean(data)
    sample_size = len(data)
    
    # Calculate the z-score
    z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
    
    # Calculate the p-value
    p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
    
    # Decision
    decision = "Reject the null hypothesis" if p_value < alpha else "Fail to reject the null hypothesis"
    
    return z_score, p_value, decision


# Example usage
if __name__ == "__main__":
    # Sample data
    data = [90, 92, 88, 85, 87, 89, 93, 91, 86]
    
    # Known population parameters
    population_mean = 85
    population_std = 5
    
    # Perform the z-test
    z_score, p_value, decision = z_test(data, population_mean, population_std)
    
    # Display results
    print(f"Z-Score: {z_score:.2f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Decision: {decision}")
