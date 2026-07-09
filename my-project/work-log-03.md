# Seaborn

1. Log-transform 
    In prediction problem, Target data often Right-Skewed
    Machine Learning model 
    Use func np.log1p(x) cause when use ln(x) with x = 0 return -inf (err)
    => When use Log-transform make data return in the form of balanced bell -> the max and min variable cant deviate the predictive path -> have prediction exactly

2. Seaborn is a Python data visualization library, based on matplotlib

3. Correclation coefficient ( Hệ số tương quan): 

4. Scatter PLot ( Biểu đồ phân tán):
    Outliers: 

5. Boxplot ( biểu đồ hộp) use for variable neighborhood ( categorical - dạng chữ)


6. Encoding: 
    Divide data to 2 specified groups:
        - Nominal ( Non order ): One-hot-encoding
        - Ordinal ( Order): Mapping
    StandardScaler: to solve problem:

7. Skewed & Scaling
    After created new variable:
        For variables (skew >0.75) -> use np.log1p()
    Scaling: