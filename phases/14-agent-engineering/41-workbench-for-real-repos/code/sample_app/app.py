"""极简注册处理器。在本练习中视为类生产环境处理。"""

USERS: dict[str, str] = {}


def signup(email: str, password: str) -> dict[str, object]:
    USERS[email] = password
    return {"status": 200, "email": email}
