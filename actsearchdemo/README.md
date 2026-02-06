This is a Next.js frontend for testing Docplus search against the local Flask API.

## Pages

- `/` password page (hardcoded client-side password)
- `/search` search interface that calls Flask `POST /search`

Hardcoded password: `actsearchdemo`

This is only a lightweight gate for demo/testing and is not secure authentication.

## Getting Started

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

UI stack includes Tailwind CSS + daisyUI.

## Vercel + local API setup

1. Deploy this Next.js app to Vercel.
2. Run the Flask API locally (`python app.py`, default `http://127.0.0.1:5000`).
3. In the search page, keep API base URL as `http://127.0.0.1:5000` (or change if needed).
4. Set Flask `DOCPLUS_ALLOWED_ORIGIN` to your Vercel domain if you want strict CORS.

Note: browser access from a Vercel-hosted page to `localhost` only works for users who have the API running on their own machine.
