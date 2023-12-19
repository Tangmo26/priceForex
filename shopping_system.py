class RegistrationSystem:
    @staticmethod
    def is_valid_username(username):
        return (8 <= len(username) <= 12) and username.isalpha() and username.isascii()

    @staticmethod
    def is_valid_password(password):
        return  any(char.isalpha() for char in password) and \
                any(char.isdigit() for char in password) and \
                8 <= len(password) <= 14


class PointsSystem:
    @staticmethod
    def calculate_points(total_price):
        if total_price < 1000:
            return 0
        else :
            return total_price // 500


class ShippingFeeSystem:
    @staticmethod
    def calculate_shipping_fee(total_price):
        if total_price >= 1000:
            return 0
        else:
            return 50


