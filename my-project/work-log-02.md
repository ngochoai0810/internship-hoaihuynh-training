# Pandas

Pandas is used for processing tabular data

2 important data types:
    - Series: 1 column
        - Create series:
            `s = pd.Series([1,3,5])`
    - DataFrame: the entire table
        - Create dataframe:
        df = pd.DataFrame({
            "Name": ["Tom", "Anna"],
            "Age": [20, 21]
        })

2. Read data:
    `df = pd.read_csv("house.csv")`
    - head(): view data in the first 5 lines
        df.head()
    - tail(): view data in the last 5 lines
        df.head()
    - column : show how many column
        df.column
    - index: index of rows
        df.index

3. describe(): show how many lines, avg mean, min, max,..
    => very useful when explore data
3.2. df.to_numpy(): Change DataFrame to Numpy 

4. Data Selection: the most important part 
    - Get a column: `df["price"]`
    - Get columns: `df[["price","bedrooms"]]`
    - Get some rows: `df[0:3]` : get 3 first rows

5. loc and iloc:
    - loc = follow name(label)
        `df.loc["house1"]` => get rows have name 'house1'
        `df.loc[:,["price"]]` => all the rows
    - iloc = follow index
        `df.iloc[1]` : get the first row
        `df.iloc[0:3, 0:2]` : get 3 first rows and 2 first col
    * loc -> label; iloc -> integer

6. Filter data:
    df[df["price"]> 20000]: get the rows have price > 20000

7. Add column:
    `df["tax"] = 5 `

8. Missing data (NaN):
    - isna(): to know which cell is mising data
        `df.isna()` = `df.isnull()`
    - dropna() : to delete these cells
        `df.dropna()`
    - fillna(): dont wanna delete, wanna to fill in replace values
        `df["Age"] = df["Age"].fillna(df["Age"].mean())`
        => fill by avg value ( like 20 NaN 25 -> 20 22.5 25)

9. Statistics:
    - `df["price"].mean()` : average
    - `df["price"].max()` : max
    - `df["price"].min()`: min
    - `df["price"].sum()`: sum

10. Groups data (Group By)
    - `df.groupby("city")["price"].sum()`:  group objects in the same city and calculate the sum of price for each city

11. Read and write file:
    `df = pd.read_csv("house.csv")`: read csv
    `df.to_csv("result.csv")` : write to csv
    `df.read_excel("house.xlsx")` : read excel
    `df.to_excel("result.xlsx")` : write to excel

12. User Defined Functions:
    
