# Internship - Agility Vietnam

**Intern:** Hoai Huynh  
**Role:** Backend & ML Integration  
**Period:** June 2026 - September 2026  
**Mentor:** Thong Nguyen

## Overview

This repository documents my internship at Agility Vietnam, including training progress and project source code.

## Authentication Flow

The auth flow for the Week 4-6 FastAPI project is:

1. The user registers with an email and password through `POST /register`.
2. The server validates the input, hashes the password, and stores only the hashed password.
3. The user logs in through `POST /login` using OAuth2 password form fields: `username` and `password`.
4. If the credentials are correct, the server signs and returns a JWT access token.
5. The Streamlit client saves that token in `st.session_state["token"]`.
6. For protected routes such as `GET /me`, the client must send the token manually in the request header:

```text
Authorization: Bearer <token>
```

The password is only used for registration and login. After login, the client proves the user's identity by sending the JWT with each protected request instead of sending the password again.

Swagger `/docs` can help with OAuth2 form login and authorization during API testing. The real Streamlit UI must handle those details itself: send login as form data, store the returned token, and attach the `Authorization` header when calling protected routes. If the header is missing, the token is invalid, or the token has expired, `/me` returns `401 Unauthorized`; this is expected behavior.

## Security Review Checklist

- Secrets such as `SECRET_KEY` are loaded from environment configuration. Do not commit a real `.env` file.
- Passwords are hashed before being saved. Routes must never store plaintext passwords.
- Response schemas are separated from input/internal schemas. Public responses use `UserResponse`, which does not include `password` or `hashed_password`.
- Auth routes declare `response_model` so FastAPI does not accidentally serialize sensitive internal fields.

## Streamlit End-to-End Test

Run the backend and Streamlit UI, then test the flow through the real UI instead of `/docs`:

1. Start the FastAPI backend.
2. Start the Streamlit app.
3. Register a new user.
4. Log in with that user.
5. Click `Call /me`.
6. Confirm the profile response shows the current user's public fields only.
7. Confirm an invalid, missing, or expired token returns `401 Unauthorized`.

## Git Merge Notes

Merge and tag are intentionally left for the repository owner to run manually:

```text
git checkout develop
git pull origin develop
git merge feature/auth-flow
git tag milestone-2-complete
git push origin develop
git push origin milestone-2-complete
```
