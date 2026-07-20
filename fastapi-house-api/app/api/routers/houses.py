from fastapi import APIRouter, status

from app.schemas.house import HouseCreate, HouseResponse


router = APIRouter(prefix="/houses", tags=["houses"])

MOCK_HOUSES = [
    HouseResponse(id=1, area=120.5, rooms=3, location="District 1", price=250000.0),
    HouseResponse(id=2, area=85.0, rooms=2, location="District 7", price=180000.0),
]


@router.get("/", response_model=list[HouseResponse])
def list_houses(limit: int = 10) -> list[HouseResponse]:
    return MOCK_HOUSES[:limit]


@router.get("/{house_id}", response_model=HouseResponse)
def get_house(house_id: int) -> HouseResponse:
    return next(
        (house for house in MOCK_HOUSES if house.id == house_id),
        MOCK_HOUSES[0],
    )


@router.post("/", response_model=HouseResponse, status_code=status.HTTP_201_CREATED)
def create_house(payload: HouseCreate) -> HouseResponse:
    return HouseResponse(id=3, price=0.0, **payload.model_dump())
