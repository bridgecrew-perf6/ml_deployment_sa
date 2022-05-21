from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class predictInput(BaseModel):
    text:str
    class Config:
        orm_mode = True

class predictCreate(BaseModel):
    text:str
