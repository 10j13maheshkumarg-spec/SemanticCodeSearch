from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/users/{user_id}")
async def get_user_profile(user_id: int):
    # Fetch user from the database
    user = fetch_user_from_db(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "name": user.name, "email": user.email}

@router.post("/auth/login")
async def authenticate_user(credentials: dict):
    username = credentials.get("username")
    password = credentials.get("password")
    
    if verify_password_hash(username, password):
        token = generate_jwt_token(username)
        return {"access_token": token, "type": "bearer"}
    return {"error": "Invalid credentials"}
