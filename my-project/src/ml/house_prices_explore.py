
import csv
from collections import defaultdict, Counter
from typing import Any, Optional, TypedDict
 
 #instead of use Dict[str, Any], we can define a TypedDict to know exactly what fields we have
class HouseRecord(TypedDict):
    id: int
    neighborhood: str
    house_style: str
    year_built: int
    lot_area: int
    overall_qual: int
    sale_price: Optional[float]  
 
#this func to read file csv and return a list of HouseRecord
def load_houses(csv_path: str, limit: Optional[int] = None) -> list[HouseRecord]:
    
    #init the list
    houses: list[HouseRecord] = []
    #open file, with will be automatically close file after finish
    with open(csv_path, newline="", encoding="utf-8") as f:
        #dictReader will read the csv file and return each row as a dict
        reader = csv.DictReader(f)
        # i : index of row, row: a dict
        for i, row in enumerate(reader):
            # if... stop. if have not it read all rows in the file
            if limit is not None and i >= limit:
                break
            #rename the row to raw
            raw: dict[str, Any] = row  
            house: HouseRecord = {
                "id": int(raw["Id"]),
                "neighborhood": raw["Neighborhood"],
                "house_style": raw["HouseStyle"],
                "year_built": int(raw["YearBuilt"]),
                "lot_area": int(raw["LotArea"]),
                "overall_qual": int(raw["OverallQual"]),
                "sale_price": float(raw["SalePrice"]) if raw.get("SalePrice") else None
            }
            # append the house to the list
            houses.append(house)
    return houses
 
 
def group_by_neighborhood(houses: list[HouseRecord]) -> dict[str, list[HouseRecord]]:
    groups: dict[str, list[HouseRecord]] = defaultdict(list)
    for house in houses:
        groups[house["neighborhood"]].append(house)
    return groups
 
#to calculate the average price by neighborhood 
def average_price_by_neighborhood(groups: dict[str, list[HouseRecord]]) -> dict[str, float]:
    return {
        name: sum(h["sale_price"] for h in hs if h["sale_price"] is not None)
        / len([h for h in hs if h["sale_price"] is not None])
        for name, hs in groups.items()
        #any return True if any element in the iterable is True. If the iterable is empty, return False.
        if any(h["sale_price"] is not None for h in hs)
    }
 
 # return top 5 house styles the most popular
def most_common_house_styles(houses: list[HouseRecord], top_n: int = 5) -> list[tuple[str, int]]:
    counter = Counter(h["house_style"] for h in houses)
    return counter.most_common(top_n)
 
 # fiter to get the expensive houses with sale_price >= min_price
def filter_expensive_houses(houses: list[HouseRecord], min_price: float) -> list[HouseRecord]:
    return [h for h in houses if h["sale_price"] is not None and h["sale_price"] >= min_price]
 
 #read file and print the summary of the data
def summarize(csv_path: str) -> None:
    houses = load_houses(csv_path)
    print(f"Total houses: {len(houses)}")
 
    groups = group_by_neighborhood(houses)
    print(f"Number of neighborhoods: {len(groups)}")
 
    avg_price = average_price_by_neighborhood(groups)
    top5_expensive = sorted(avg_price.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 neighborhoods with the highest average prices:")
    for name, price in top5_expensive:
        print(f"  {name}: ${price:,.0f}")
 
    print("\nTop 5 most common house styles:")
    for style, count in most_common_house_styles(houses):
        print(f"  {style}: {count} houses")
 
    expensive = filter_expensive_houses(houses, min_price=300_000)
    print(f"\nNumber of houses priced >= $300,000: {len(expensive)}")
 
 
if __name__ == "__main__":
    summarize("data/raw/train.csv")