from fastapi import APIRouter, HTTPException, status

from app.schemas.house import HouseCreate, HouseResponse, HouseUpdate


router = APIRouter(prefix="/houses", tags=["houses"])

MOCK_HOUSES = {
    1: {"id": 1, "area": 120.5, "rooms": 3, "location": "District 1", "price": 250000.0},
    2: {"id": 2, "area": 85.0, "rooms": 2, "location": "District 7", "price": 180000.0},
}


@router.get("/", response_model=list[HouseResponse])
def list_houses(limit: int = 10) -> list[dict[str, object]]:
    return list(MOCK_HOUSES.values())[:limit]


@router.get("/{house_id}", response_model=HouseResponse)
def get_house(house_id: int) -> dict[str, object]:
    house = MOCK_HOUSES.get(house_id)
    if house is None:
        raise HTTPException(status_code=404, detail="House not found")
    return house


@router.post("/", response_model=HouseResponse, status_code=status.HTTP_201_CREATED)
def create_house(payload: HouseCreate) -> dict[str, object]:
    return {"id": 3, "price": 0.0, **payload.model_dump()}


@router.patch("/{house_id}", response_model=HouseResponse)
def update_house(house_id: int, payload: HouseUpdate) -> dict[str, object]:
    house = MOCK_HOUSES.get(house_id)
    if house is None:
        raise HTTPException(status_code=404, detail="House not found")

    update_data = payload.model_dump(exclude_unset=True)
    house.update(update_data)
    return house
