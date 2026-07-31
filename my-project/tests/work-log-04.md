# Core Areas Of Knowledge

1. Modularity and SRP
   - Avoid writing one large function for the whole data pipeline.
   - Split the pipeline into independent functions with one clear responsibility.
   - Keep source code under `src/ml` and tests under `tests`.

2. Constant Management
   - Avoid hard-coded repeated values.
   - Keep shared column groups such as missing-value strategies in named constants.

3. Type Hinting and Static Checking
   - Use explicit typing such as `list[str]`, `dict[str, int]`, and
     `pd.DataFrame -> pd.DataFrame`.
   - Use `mypy` to catch type-related mistakes before runtime.

4. Immutability To Avoid Side Effects
   - Copy the input DataFrame before modifying it, for example
     `df_clean = df.copy()`.
   - Avoid changing the caller's original data because hidden mutation makes
     debugging harder.

5. Docstrings
   - Write short docstrings that explain what a function does.
   - Include parameters, return values, and expected errors when the function is
     part of the public workflow.

6. Unit Testing
   - Use small mock datasets with `pytest.fixture`.
   - Test calculation logic, immutability, error handling, and pipeline
     consistency.
