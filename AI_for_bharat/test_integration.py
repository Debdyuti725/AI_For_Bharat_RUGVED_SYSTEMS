"""Test the full integrated flow: signup -> profile -> eligibility -> chat"""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8000"

def api(method, path, body=None, token=None):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": json.loads(e.read())}

# ─── 1. Sign Up ───
print("=" * 50)
print("  Step 1: Sign Up")
result = api("POST", "/api/auth/signup", {"email": "vedh@test.com", "password": "test1234", "name": "Vedh Sontha"})
if "error" in result:
    # Already exists, try login
    print("  Account exists, logging in...")
    result = api("POST", "/api/auth/login", {"email": "vedh@test.com", "password": "test1234"})
token = result["token"]
print("  Token:", token[:30] + "...")
print("  User ID:", result["user_id"])

# ─── 2. Check /me ───
print("\n  Step 2: Check /api/auth/me")
me = api("GET", "/api/auth/me", token=token)
print("  Logged in as:", me.get("name", me.get("email")))

# ─── 3. Submit Tier 1 Profile ───
print("\n  Step 3: Submit Tier 1 Profile")
profile_result = api("POST", "/api/profile", {
    "name": "Vedh Sontha",
    "age": 19,
    "gender": "male",
    "state": "karnataka",
    "area_type": "urban",
    "category": "general",
    "education_level": "graduate",
    "employment_status": "student",
    "annual_income": 800000,
    "family_size": 3,
}, token=token)
print("  Profile saved! Completion:", profile_result.get("tier1_completion", "?"))

# ─── 4. Get Eligibility ───
print("\n  Step 4: Get Eligibility Scores")
scores = api("GET", "/api/eligibility?top_n=5", token=token)
if isinstance(scores, list):
    for s in scores:
        print("  [%5.1f%%] %s" % (s["eligibility_score"], s["scheme_name"]))
else:
    print("  Error:", scores)

# ─── 5. Get Saved Profile ───
print("\n  Step 5: Get Saved Profile from DB")
saved = api("GET", "/api/profile", token=token)
print("  Tier 1 complete:", saved.get("tier1_complete"))
print("  Profile name:", saved.get("profile", {}).get("name", "?"))

# ─── 6. Test without token (should fail) ───
print("\n  Step 6: Test without auth (expect 401)")
noauth = api("GET", "/api/eligibility", token=None)
print("  Got error:", noauth.get("error", "none"), noauth.get("detail", {}).get("detail", ""))

print("\n" + "=" * 50)
print("  ALL INTEGRATION TESTS DONE!")
print("=" * 50)
