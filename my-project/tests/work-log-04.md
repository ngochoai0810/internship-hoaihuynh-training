# Core areas of knowledge

1. Modularity ( Tư duy Module hóa) & SPR ( Nguyên tắc Trách nhiệm đơn lẻ):
    - Don't write one huge function to processing all the data
    - Need to split pipeline to independent func, 1 func = 1 feature
    - Understand the standard directory structure, clearly about main source code `scr/ml` and test code `test/`

2. Constant Management
    - No hardcode: Need to 
    - 

3. Type Hinting & Static Checking
    - Clearly define: understand how to use typing ( List, Dict) và Type Hinting of Pandas ( df: pd.DataFrame -> pd.DataFrame)
    - Proactive catch error: Understand benefit of using tools `mypy` before run code

4. Immutability ( Tính bất biến) to avoid Side Effects
    - Protect origin data: Understand the rule : alway init df_clean = df.copy()
    - Reason: If u fix direct origin data, debug work become hard 

5. Docstring ( Tiêu chuẩn hóa tài liệu)
    - Write code to others read: Understand how to write standard Docstring ( Google Style & NumPy Style)
    - Force Structure: A gud docstring need to describe this func work for ?, parameter, return what?, and any error can happen

6. Unit Testing:
    - Mock Data: Know to create the small dataset ( though @pytest.fixture) 
    - Write test case: Understand to use library pytest and keyword assert to:
        - Checking caculation logic isnt true ?
        - Check function is immutability?
    