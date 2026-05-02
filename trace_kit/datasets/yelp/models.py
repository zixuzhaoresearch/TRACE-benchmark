"""Data models for Yelp Academic Dataset."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class YelpReview:
    """A single Yelp review."""
    review_id: str
    business_id: str
    stars: float
    text: str
    date: str
    useful: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'review_id': self.review_id,
            'business_id': self.business_id,
            'stars': self.stars,
            'text': self.text,
            'date': self.date,
            'useful': self.useful,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'YelpReview':
        return cls(
            review_id=data['review_id'],
            business_id=data['business_id'],
            stars=data['stars'],
            text=data['text'],
            date=data['date'],
            useful=data.get('useful', 0),
        )


@dataclass
class YelpPOI:
    """A Yelp business (Point of Interest) with its reviews."""
    business_id: str
    name: str
    city: str
    state: str
    poi_type: str  # 'restaurant', 'hotel', 'attraction'
    address: str
    stars: float
    review_count: int
    categories: List[str]
    latitude: float
    longitude: float
    price_range: Optional[str] = None  # '$', '$$', '$$$', '$$$$'
    attributes: Dict[str, Any] = field(default_factory=dict)
    reviews: List[YelpReview] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'business_id': self.business_id,
            'name': self.name,
            'city': self.city,
            'state': self.state,
            'poi_type': self.poi_type,
            'address': self.address,
            'stars': self.stars,
            'review_count': self.review_count,
            'categories': self.categories,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'price_range': self.price_range,
            'attributes': self.attributes,
            'reviews': [r.to_dict() for r in self.reviews],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'YelpPOI':
        reviews = [YelpReview.from_dict(r) for r in data.get('reviews', [])]
        return cls(
            business_id=data['business_id'],
            name=data['name'],
            city=data['city'],
            state=data.get('state', ''),
            poi_type=data['poi_type'],
            address=data['address'],
            stars=data['stars'],
            review_count=data['review_count'],
            categories=data['categories'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            price_range=data.get('price_range'),
            attributes=data.get('attributes', {}),
            reviews=reviews,
        )

    @classmethod
    def from_yelp_business(cls, biz: Dict[str, Any], poi_type: str) -> 'YelpPOI':
        """Create from raw Yelp business JSON record."""
        categories = []
        if biz.get('categories'):
            categories = [c.strip() for c in biz['categories'].split(',')]

        # Extract price range from attributes
        price_range = None
        attrs = biz.get('attributes') or {}
        if isinstance(attrs, dict):
            price_val = attrs.get('RestaurantsPriceRange2')
            if price_val:
                try:
                    price_range = '$' * int(price_val)
                except (ValueError, TypeError):
                    pass

        # Clean attributes: keep only useful ones
        useful_attrs = {}
        attr_keys = [
            'WiFi', 'HasTV', 'OutdoorSeating', 'RestaurantsReservations',
            'RestaurantsDelivery', 'RestaurantsTakeOut', 'GoodForKids',
            'BikeParking', 'Alcohol', 'NoiseLevel', 'Ambience',
            'GoodForMeal', 'BusinessParking',
        ]
        for key in attr_keys:
            if key in attrs:
                useful_attrs[key] = attrs[key]

        return cls(
            business_id=biz['business_id'],
            name=biz['name'],
            city=biz['city'],
            state=biz.get('state', ''),
            poi_type=poi_type,
            address=biz.get('address', ''),
            stars=biz['stars'],
            review_count=biz['review_count'],
            categories=categories,
            latitude=biz.get('latitude', 0.0),
            longitude=biz.get('longitude', 0.0),
            price_range=price_range,
            attributes=useful_attrs,
        )
