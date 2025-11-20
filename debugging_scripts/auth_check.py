import earthaccess
import os

def check_nasa_auth():
    try:
        # Try to login
        auth = earthaccess.login()
        if auth:
            print("✅ NASA authentication successful!")
            return True
        else:
            print("❌ NASA authentication failed!")
            print("Please run: python -c 'import earthaccess; earthaccess.login()'")
            return False
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        print("Please set up NASA Earthdata credentials first!")
        return False

if __name__ == "__main__":
    check_nasa_auth()
