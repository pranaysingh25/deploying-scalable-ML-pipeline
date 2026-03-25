"""
Data Validation Schemas Module

This module defines Pydantic models for request/response validation in the API.
It ensures type safety and automatic documentation generation for API endpoints.

Classes:
    CensusData: Pydantic model for census income prediction request data
"""

from pydantic import BaseModel, Field, ConfigDict

class CensusData(BaseModel):
    """
    Pydantic model for census data used in income prediction.
    
    Represents an individual's census information used to predict income level.
    All field names use hyphens in API requests (e.g., 'education-num') but are
    converted to underscores internally (e.g., 'education_num') using Pydantic aliases.
    
    Attributes
    ----------
    age : int
        Age of the individual
    workclass : str
        Employment sector (e.g., Private, Self-emp, Federal-gov, etc.)
    fnlgt : int
        Final weight - census sampling weight
    education : str
        Highest level of education attained
    education_num : int
        Numeric encoding of education level
    marital_status : str
        Marital status (Married-civ-spouse, Never-married, Divorced, etc.)
    occupation : str
        Type of occupation
    relationship : str
        Relationship to head of household
    race : str
        Racial category
    sex : str
        Gender (Male or Female)
    capital_gain : int
        Capital gains from investments
    capital_loss : int
        Capital losses from investments
    hours_per_week : int
        Hours worked per week
    native_country : str
        Country of origin
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }
    )

