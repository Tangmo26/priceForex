import pytest
from shopping_system import RegistrationSystem, PointsSystem, ShippingFeeSystem

# 1. Registration System Tests
def test_valid_username() :
    # 1.1 leght of 8 - 12
    assert RegistrationSystem.is_valid_username("abcdefgh") is True
    assert RegistrationSystem.is_valid_username("abcdefghijkldaasd") is False
    
    # 1.2 charactor English only
    assert RegistrationSystem.is_valid_username("abcasddas") is True
    assert RegistrationSystem.is_valid_username("dasหก;54กd") is False
    
    # 1.3 password must have letter and number contain least 1 charactor
    assert RegistrationSystem.is_valid_password("dasdad6565") is True
    assert RegistrationSystem.is_valid_password("dasdasdda") is False
    
    # 1.4 password must leght of 8 - 14
    assert RegistrationSystem.is_valid_password("dasdwdad54") is True
    assert RegistrationSystem.is_valid_password("asldkallas5dawdasdwda") is False

# 2. Points System Tests
def test_points_accumulation() :
    # 2.1 product with a price less than 1000
    assert PointsSystem.calculate_points(500) == 0
    
    # 2.2 product with a price of 1000 more you will get 1 point any 500 baht
    assert PointsSystem.calculate_points(999) == 0
    assert PointsSystem.calculate_points(2500) == 5
    assert PointsSystem.calculate_points(3000) == 6

# 3. Shipping Fee System Tests
def test_shipping_fee():
    # 3.1 not calculate free value when buy product with a price of 1000 more
    assert ShippingFeeSystem.calculate_shipping_fee(1500) == 0
    
    #3.2 calculate free value when buy product with a price of less than 1000
    assert ShippingFeeSystem.calculate_shipping_fee(900) == 50


