from src.ml import predict_churn

def main():
    result = predict_churn()
    print(f"Result from ML Layer: {result}")
    
    print("Successfully")

if __name__ == "__main__":
    main()