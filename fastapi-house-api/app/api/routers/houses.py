from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.schemas.house import HouseCreate, HouseResponse, HouseUpdate
from orm_migrations.src.database import get_db
from orm_migrations.src.models import House


router = APIRouter(prefix="/houses", tags=["houses"])


@router.get("", response_model=list[HouseResponse])
def list_houses(
    limit: int = Query(default=10, gt=0),
    min_price: float | None = Query(default=None, ge=0),
    max_rooms: int | None = Query(default=None, gt=0),
    db: Session = Depends(get_db),
) -> list[House]:
    statement = select(House).order_by(House.id)
    if min_price is not None:
        statement = statement.where(House.price >= min_price)
    if max_rooms is not None:
        statement = statement.where(House.rooms <= max_rooms)

    return list(db.scalars(statement.limit(limit)))


@router.get("/{house_id}", response_model=HouseResponse)
def get_house(house_id: int, db: Session = Depends(get_db)) -> House:
    house = db.get(House, house_id)
    if house is None:
        raise HTTPException(status_code=404, detail="House not found")
    return house


@router.post("", response_model=HouseResponse, status_code=status.HTTP_201_CREATED)
def create_house(payload: HouseCreate, db: Session = Depends(get_db)) -> House:
    house = House(price=0.0, **payload.model_dump())
    db.add(house)
    db.commit()
    db.refresh(house)
    return house


@router.patch("/{house_id}", response_model=HouseResponse)
def update_house(
    house_id: int,
    payload: HouseUpdate,
    db: Session = Depends(get_db),
) -> House:
    house = db.get(House, house_id)
    if house is None:
        raise HTTPException(status_code=404, detail="House not found")

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(house, field, value)

    db.commit()
    db.refresh(house)
    return house
